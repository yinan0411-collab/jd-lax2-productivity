from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import streamlit as st

try:
    import python_calamine  # noqa: F401

    HAS_CALAMINE = True
except ImportError:
    HAS_CALAMINE = False


APP_TITLE = "LAX2 人效分析工具"

EMPLOYEE_COLUMN_ALIASES = {
    "user_id": ["用户编码", "Use ID", "UseID", "User ID", "userid", "use id"],
    "erp": ["ERP", "erp"],
    "name": ["姓名", "员工姓名", "Name", "Employee Name"],
}

ACCEPTANCE_COLUMNS = {
    "pi": "京东入库单号",
    "quantity": "验收量",
    "operator": "验收人",
    "start": "开始验收时间",
    "end": "最后验收时间",
}

PICKING_COLUMNS = {
    "order": "订单号",
    "task": "任务单号",
    "area": "所属储区",
    "quantity": "实际拣货量",
    "receive": "任务领取时间",
    "finish": "拣货完成时间",
    "employee_id": "工号",
    "email": "邮箱",
}


@dataclass(frozen=True)
class EmployeeLookup:
    exact_user_id: dict[str, str]
    exact_erp: dict[str, str]
    normalized_user_id: dict[str, str]
    normalized_erp: dict[str, str]
    duplicate_keys: list[str]
    source_rows: int


@dataclass(frozen=True)
class AcceptanceResult:
    ranking: pd.DataFrame
    total_pieces: float
    total_orders: int
    total_system_seconds: float
    total_actual_seconds: float
    overall_piece_ratio: float
    overall_hours_per_order: float
    operator_count: int
    valid_row_count: int
    total_row_count: int
    excluded_unmatched_rows: int
    invalid_row_count: int


@dataclass(frozen=True)
class PickingResult:
    ranking: pd.DataFrame
    file_summary: pd.DataFrame
    total_pieces: float
    total_orders: int
    total_tasks: int
    total_actual_seconds: float
    overall_task_piece_ratio: float
    overall_hourly_orders: float
    overall_hourly_tasks: float
    operator_count: int
    uploaded_row_count: int
    deduplicated_row_count: int
    valid_row_count: int
    r_area_row_count: int
    missing_time_row_count: int
    excluded_unmatched_rows: int


def normalize_column_label(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def find_column(columns: Iterable[object], aliases: Sequence[str]) -> str | None:
    normalized_to_original = {
        normalize_column_label(column): str(column) for column in columns
    }
    for alias in aliases:
        matched = normalized_to_original.get(normalize_column_label(alias))
        if matched is not None:
            return matched
    return None


def canonical_key(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().casefold()


def compact_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", canonical_key(value))


def clean_name(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def clean_identifier(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.casefold() in {"nan", "none", "nat"}:
        return ""
    return text


def extract_operator_id(value: object) -> str:
    """Convert a system account/email value into the ID used by the employee master."""
    if value is None or pd.isna(value):
        return ""

    text = str(value).strip()
    parenthetical_values = re.findall(r"\(([^()]*)\)", text)
    if parenthetical_values:
        candidate = next(
            (item for item in reversed(parenthetical_values) if "@" in item),
            parenthetical_values[-1],
        )
    else:
        candidate = text

    candidate = candidate.strip()
    if "@" in candidate:
        candidate = candidate.split("@", 1)[0]
    return candidate.strip()


def choose_picking_operator_id(employee_id: object, email: object) -> str:
    employee = extract_operator_id(employee_id)
    if employee:
        return employee
    return extract_operator_id(email)


def _add_lookup_value(
    target: dict[str, str],
    key: str,
    name: str,
    duplicate_keys: list[str],
    key_label: str,
) -> None:
    if not key:
        return
    if key in target and target[key] != name:
        duplicate_keys.append(f"{key_label}: {key}")
        return
    target.setdefault(key, name)


def build_employee_lookup(employee_df: pd.DataFrame) -> EmployeeLookup:
    user_id_col = find_column(employee_df.columns, EMPLOYEE_COLUMN_ALIASES["user_id"])
    erp_col = find_column(employee_df.columns, EMPLOYEE_COLUMN_ALIASES["erp"])
    name_col = find_column(employee_df.columns, EMPLOYEE_COLUMN_ALIASES["name"])

    if name_col is None:
        raise ValueError(
            "人员源数据中未找到姓名列。支持：姓名、员工姓名、Name、Employee Name。"
        )
    if user_id_col is None and erp_col is None:
        raise ValueError("人员源数据中未找到用户编码/Use ID或ERP列。")

    exact_user_id: dict[str, str] = {}
    exact_erp: dict[str, str] = {}
    normalized_user_id: dict[str, str] = {}
    normalized_erp: dict[str, str] = {}
    duplicate_keys: list[str] = []

    for _, row in employee_df.iterrows():
        name = clean_name(row.get(name_col))
        if not name:
            continue

        if user_id_col is not None:
            raw_user_id = row.get(user_id_col)
            _add_lookup_value(
                exact_user_id,
                canonical_key(raw_user_id),
                name,
                duplicate_keys,
                "用户编码",
            )
            _add_lookup_value(
                normalized_user_id,
                compact_key(raw_user_id),
                name,
                duplicate_keys,
                "标准化用户编码",
            )

        if erp_col is not None:
            raw_erp = row.get(erp_col)
            _add_lookup_value(
                exact_erp,
                canonical_key(raw_erp),
                name,
                duplicate_keys,
                "ERP",
            )
            _add_lookup_value(
                normalized_erp,
                compact_key(raw_erp),
                name,
                duplicate_keys,
                "标准化ERP",
            )

    return EmployeeLookup(
        exact_user_id=exact_user_id,
        exact_erp=exact_erp,
        normalized_user_id=normalized_user_id,
        normalized_erp=normalized_erp,
        duplicate_keys=sorted(set(duplicate_keys)),
        source_rows=len(employee_df),
    )


def match_employee_name(operator_id: str, lookup: EmployeeLookup | None) -> str:
    if lookup is None:
        return ""

    exact = canonical_key(operator_id)
    compact = compact_key(operator_id)

    if exact in lookup.exact_user_id:
        return lookup.exact_user_id[exact]
    if exact in lookup.exact_erp:
        return lookup.exact_erp[exact]
    if compact and compact in lookup.normalized_user_id:
        return lookup.normalized_user_id[compact]
    if compact and compact in lookup.normalized_erp:
        return lookup.normalized_erp[compact]
    return ""


def format_duration(total_seconds: float) -> str:
    rounded = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def split_interval_by_day(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[object, pd.Timestamp, pd.Timestamp]]:
    """Split an interval at midnight so overlap removal is performed separately by day."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return []

    pieces: list[tuple[object, pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    while cursor.date() < end_ts.date():
        midnight = cursor.normalize() + pd.Timedelta(days=1)
        pieces.append((cursor.date(), cursor, midnight))
        cursor = midnight

    pieces.append((cursor.date(), cursor, end_ts))
    return pieces


def merge_interval_seconds(intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> float:
    cleaned = sorted(
        [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in intervals if end >= start],
        key=lambda item: item[0],
    )
    if not cleaned:
        return 0.0

    current_start, current_end = cleaned[0]
    total = 0.0
    for next_start, next_end in cleaned[1:]:
        if next_start <= current_end:
            current_end = max(current_end, next_end)
        else:
            total += (current_end - current_start).total_seconds()
            current_start, current_end = next_start, next_end
    total += (current_end - current_start).total_seconds()
    return max(total, 0.0)


def calculate_daily_merged_seconds(
    interval_df: pd.DataFrame,
    operator_col: str,
    start_col: str,
    end_col: str,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in interval_df[[operator_col, start_col, end_col]].itertuples(index=False, name=None):
        operator_id, start, end = row
        for work_date, part_start, part_end in split_interval_by_day(start, end):
            records.append(
                {
                    "账号": operator_id,
                    "工作日期": work_date,
                    "开始": part_start,
                    "结束": part_end,
                }
            )

    if not records:
        return pd.DataFrame(columns=["账号", "秒数"])

    split_df = pd.DataFrame(records)
    daily_records: list[dict[str, object]] = []
    for (operator_id, work_date), group in split_df.groupby(["账号", "工作日期"], sort=False):
        daily_records.append(
            {
                "账号": operator_id,
                "工作日期": work_date,
                "秒数": merge_interval_seconds(zip(group["开始"], group["结束"])),
            }
        )

    daily = pd.DataFrame(daily_records)
    return daily.groupby("账号", as_index=False)["秒数"].sum()


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes, file_name: str) -> list[str]:
    engine = "calamine" if HAS_CALAMINE else (
        "xlrd" if Path(file_name).suffix.lower() == ".xls" else "openpyxl"
    )
    with pd.ExcelFile(io.BytesIO(file_bytes), engine=engine) as workbook:
        return workbook.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_sheet(
    file_bytes: bytes,
    file_name: str,
    sheet_name: str | int,
    usecols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    engine = "calamine" if HAS_CALAMINE else (
        "xlrd" if Path(file_name).suffix.lower() == ".xls" else "openpyxl"
    )
    selected_columns = list(usecols) if usecols else None
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        usecols=selected_columns,
        engine=engine,
    )


def calculate_acceptance_productivity(
    acceptance_df: pd.DataFrame,
    employee_lookup: EmployeeLookup | None,
) -> AcceptanceResult:
    required = list(ACCEPTANCE_COLUMNS.values())
    missing = [column for column in required if column not in acceptance_df.columns]
    if missing:
        raise ValueError("验收表缺少必要字段：" + "、".join(missing))

    work = acceptance_df[required].copy()
    total_row_count = len(work)
    work["验收人账号"] = work[ACCEPTANCE_COLUMNS["operator"]].map(extract_operator_id)
    work["实际姓名"] = work["验收人账号"].map(
        lambda value: match_employee_name(value, employee_lookup)
    )
    work["PI"] = work[ACCEPTANCE_COLUMNS["pi"]].map(clean_identifier)
    work["验收件量数值"] = pd.to_numeric(
        work[ACCEPTANCE_COLUMNS["quantity"]], errors="coerce"
    )
    work["开始时间"] = pd.to_datetime(work[ACCEPTANCE_COLUMNS["start"]], errors="coerce")
    work["结束时间"] = pd.to_datetime(work[ACCEPTANCE_COLUMNS["end"]], errors="coerce")

    basic_valid = (
        work["验收人账号"].ne("")
        & work["PI"].ne("")
        & work["验收件量数值"].notna()
        & work["开始时间"].notna()
        & work["结束时间"].notna()
        & (work["结束时间"] >= work["开始时间"])
    )
    invalid_row_count = int((~basic_valid).sum())
    work = work.loc[basic_valid].copy()

    excluded_unmatched_rows = int(work["实际姓名"].eq("").sum())
    valid = work.loc[work["实际姓名"].ne("")].copy()
    if valid.empty:
        raise ValueError("没有匹配到人员主数据的有效验收记录。")

    # System-operation duration: merge all raw operation intervals by employee and day.
    system_duration = calculate_daily_merged_seconds(
        valid, "验收人账号", "开始时间", "结束时间"
    ).rename(columns={"账号": "验收人账号", "秒数": "系统操作秒数"})

    # Actual acceptance duration: create one interval for each employee + day + PI,
    # then remove overlaps among PI intervals within the same day.
    valid["PI日期"] = valid["开始时间"].dt.date
    pi_intervals = (
        valid.groupby(["验收人账号", "PI日期", "PI"], as_index=False)
        .agg(PI开始时间=("开始时间", "min"), PI结束时间=("结束时间", "max"))
    )
    actual_duration = calculate_daily_merged_seconds(
        pi_intervals, "验收人账号", "PI开始时间", "PI结束时间"
    ).rename(columns={"账号": "验收人账号", "秒数": "实际验收秒数"})

    summary = (
        valid.groupby(["验收人账号", "实际姓名"], as_index=False)
        .agg(
            验收件量=("验收件量数值", "sum"),
            验收单量=("PI", "nunique"),
        )
    )
    summary = summary.merge(system_duration, on="验收人账号", how="left")
    summary = summary.merge(actual_duration, on="验收人账号", how="left")
    summary[["系统操作秒数", "实际验收秒数"]] = summary[
        ["系统操作秒数", "实际验收秒数"]
    ].fillna(0.0)

    summary["单件比"] = np.where(
        summary["验收单量"] > 0,
        summary["验收件量"] / summary["验收单量"],
        np.nan,
    )
    summary["人效（小时单量）"] = np.where(
        summary["验收单量"] > 0,
        (summary["实际验收秒数"] / 3600) / summary["验收单量"],
        np.nan,
    )
    summary["系统操作时长"] = summary["系统操作秒数"].map(format_duration)
    summary["实际验收时长"] = summary["实际验收秒数"].map(format_duration)

    summary = summary.sort_values(
        ["人效（小时单量）", "验收件量"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "排名", np.arange(1, len(summary) + 1))

    ranking = summary[
        [
            "排名",
            "验收人账号",
            "实际姓名",
            "验收件量",
            "验收单量",
            "单件比",
            "系统操作时长",
            "实际验收时长",
            "人效（小时单量）",
        ]
    ].copy()

    total_pieces = float(valid["验收件量数值"].sum())
    total_orders = int(valid["PI"].nunique())
    total_system_seconds = float(summary["系统操作秒数"].sum())
    total_actual_seconds = float(summary["实际验收秒数"].sum())
    overall_piece_ratio = total_pieces / total_orders if total_orders else float("nan")
    overall_hours_per_order = (
        (total_actual_seconds / 3600) / total_orders if total_orders else float("nan")
    )

    return AcceptanceResult(
        ranking=ranking,
        total_pieces=total_pieces,
        total_orders=total_orders,
        total_system_seconds=total_system_seconds,
        total_actual_seconds=total_actual_seconds,
        overall_piece_ratio=overall_piece_ratio,
        overall_hours_per_order=overall_hours_per_order,
        operator_count=len(summary),
        valid_row_count=len(valid),
        total_row_count=total_row_count,
        excluded_unmatched_rows=excluded_unmatched_rows,
        invalid_row_count=invalid_row_count,
    )


def combine_picking_files(uploaded_files: Sequence[object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    required = tuple(PICKING_COLUMNS.values())

    for uploaded in uploaded_files:
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        try:
            frame = read_excel_sheet(file_bytes, file_name, 0, required)
        except ValueError as exc:
            raise ValueError(f"文件“{file_name}”缺少必要字段或读取失败：{exc}") from exc

        original_rows = len(frame)
        frame["来源文件"] = file_name
        frames.append(frame)
        summary_rows.append({"文件名": file_name, "读取行数": original_rows})

    if not frames:
        raise ValueError("请至少上传一个拣货文件。")

    combined = pd.concat(frames, ignore_index=True)
    file_summary = pd.DataFrame(summary_rows)
    return combined, file_summary


def calculate_picking_productivity(
    combined_df: pd.DataFrame,
    file_summary: pd.DataFrame,
    employee_lookup: EmployeeLookup | None,
) -> PickingResult:
    required = list(PICKING_COLUMNS.values())
    missing = [column for column in required if column not in combined_df.columns]
    if missing:
        raise ValueError("拣货表缺少必要字段：" + "、".join(missing))

    uploaded_row_count = len(combined_df)

    # Remove exact duplicates across weekly files. Source file is excluded from the key.
    work = combined_df.drop_duplicates(subset=required, keep="first").copy()
    deduplicated_row_count = len(work)

    work["所属储区标准值"] = (
        work[PICKING_COLUMNS["area"]].fillna("").astype(str).str.strip().str.upper()
    )
    r_area_mask = work["所属储区标准值"].eq("R")
    r_area_row_count = int(r_area_mask.sum())
    work = work.loc[~r_area_mask].copy()

    work["拣货人账号"] = [
        choose_picking_operator_id(employee_id, email)
        for employee_id, email in zip(
            work[PICKING_COLUMNS["employee_id"]],
            work[PICKING_COLUMNS["email"]],
        )
    ]
    work["实际姓名"] = work["拣货人账号"].map(
        lambda value: match_employee_name(value, employee_lookup)
    )
    work["订单号标准值"] = work[PICKING_COLUMNS["order"]].map(clean_identifier)
    work["任务单号标准值"] = work[PICKING_COLUMNS["task"]].map(clean_identifier)
    work["拣货件量数值"] = pd.to_numeric(
        work[PICKING_COLUMNS["quantity"]], errors="coerce"
    )
    work["领取时间"] = pd.to_datetime(
        work[PICKING_COLUMNS["receive"]], errors="coerce"
    )
    work["完成时间"] = pd.to_datetime(
        work[PICKING_COLUMNS["finish"]], errors="coerce"
    )

    has_time = work["领取时间"].notna() & work["完成时间"].notna()
    missing_time_row_count = int((~has_time).sum())

    valid_mask = (
        work["拣货人账号"].ne("")
        & work["订单号标准值"].ne("")
        & work["任务单号标准值"].ne("")
        & work["拣货件量数值"].notna()
        & has_time
        & (work["完成时间"] >= work["领取时间"])
    )
    work = work.loc[valid_mask].copy()

    excluded_unmatched_rows = int(work["实际姓名"].eq("").sum())
    valid = work.loc[work["实际姓名"].ne("")].copy()
    if valid.empty:
        raise ValueError("排除R区、无领取时间和未匹配人员后，没有可计算的拣货记录。")

    # Build one complete interval per employee + day + picking task.
    valid["任务日期"] = valid["领取时间"].dt.date
    task_intervals = (
        valid.groupby(["拣货人账号", "任务日期", "任务单号标准值"], as_index=False)
        .agg(
            任务开始时间=("领取时间", "min"),
            任务结束时间=("完成时间", "max"),
        )
    )
    actual_duration = calculate_daily_merged_seconds(
        task_intervals, "拣货人账号", "任务开始时间", "任务结束时间"
    ).rename(columns={"账号": "拣货人账号", "秒数": "实际拣货秒数"})

    summary = (
        valid.groupby(["拣货人账号", "实际姓名"], as_index=False)
        .agg(
            拣货件量=("拣货件量数值", "sum"),
            订单量=("订单号标准值", "nunique"),
            任务单量=("任务单号标准值", "nunique"),
        )
    )
    summary = summary.merge(actual_duration, on="拣货人账号", how="left")
    summary["实际拣货秒数"] = summary["实际拣货秒数"].fillna(0.0)
    summary["实际拣货小时"] = summary["实际拣货秒数"] / 3600
    summary["任务件比"] = np.where(
        summary["任务单量"] > 0,
        summary["拣货件量"] / summary["任务单量"],
        np.nan,
    )
    summary["人效（小时单量）"] = np.where(
        summary["实际拣货小时"] > 0,
        summary["订单量"] / summary["实际拣货小时"],
        np.nan,
    )
    summary["小时任务单量"] = np.where(
        summary["实际拣货小时"] > 0,
        summary["任务单量"] / summary["实际拣货小时"],
        np.nan,
    )
    summary["实际拣货时长"] = summary["实际拣货秒数"].map(format_duration)

    summary = summary.sort_values(
        ["人效（小时单量）", "小时任务单量", "拣货件量"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "排名", np.arange(1, len(summary) + 1))

    ranking = summary[
        [
            "排名",
            "拣货人账号",
            "实际姓名",
            "拣货件量",
            "订单量",
            "任务单量",
            "任务件比",
            "实际拣货时长",
            "人效（小时单量）",
            "小时任务单量",
        ]
    ].copy()

    total_pieces = float(valid["拣货件量数值"].sum())
    total_orders = int(valid["订单号标准值"].nunique())
    total_tasks = int(valid["任务单号标准值"].nunique())
    total_actual_seconds = float(summary["实际拣货秒数"].sum())
    total_hours = total_actual_seconds / 3600

    return PickingResult(
        ranking=ranking,
        file_summary=file_summary,
        total_pieces=total_pieces,
        total_orders=total_orders,
        total_tasks=total_tasks,
        total_actual_seconds=total_actual_seconds,
        overall_task_piece_ratio=(total_pieces / total_tasks if total_tasks else float("nan")),
        overall_hourly_orders=(total_orders / total_hours if total_hours else float("nan")),
        overall_hourly_tasks=(total_tasks / total_hours if total_hours else float("nan")),
        operator_count=len(summary),
        uploaded_row_count=uploaded_row_count,
        deduplicated_row_count=deduplicated_row_count,
        valid_row_count=len(valid),
        r_area_row_count=r_area_row_count,
        missing_time_row_count=missing_time_row_count,
        excluded_unmatched_rows=excluded_unmatched_rows,
    )


def _excel_formats(workbook):
    return {
        "header": workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        ),
        "integer": workbook.add_format({"num_format": "#,##0"}),
        "decimal": workbook.add_format({"num_format": "#,##0.00"}),
        "decimal4": workbook.add_format({"num_format": "0.0000"}),
        "text": workbook.add_format({"valign": "vcenter"}),
    }


def make_acceptance_excel(result: AcceptanceResult) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="验收人效排名", index=False)
        method = pd.DataFrame(
            {
                "项目": [
                    "验收件量",
                    "验收单量",
                    "单件比",
                    "系统操作时长",
                    "实际验收时长",
                    "人效（小时单量）",
                    "人员范围",
                ],
                "计算口径": [
                    "按人员汇总验收量",
                    "京东入库单号去重数量",
                    "验收件量 ÷ 验收单量",
                    "原始系统操作区间按人员、按自然日去除重叠后相加",
                    "单个PI取最早开始至最晚结束；同一人员同一天的PI区间去除重叠后相加",
                    "实际验收小时 ÷ 验收单量（小时/单，越低表示平均每单用时越短）",
                    "仅保留成功匹配人员主数据的员工",
                ],
            }
        )
        method.to_excel(writer, sheet_name="计算口径", index=False)

        workbook = writer.book
        fmt = _excel_formats(workbook)
        sheet = writer.sheets["验收人效排名"]
        sheet.freeze_panes(1, 0)
        sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        sheet.set_row(0, 24, fmt["header"])
        widths = [8, 22, 22, 14, 14, 12, 18, 18, 18]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(0, 0, 8, fmt["integer"])
        sheet.set_column(3, 4, 14, fmt["integer"])
        sheet.set_column(5, 5, 12, fmt["decimal"])
        sheet.set_column(8, 8, 18, fmt["decimal4"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 100)
    return output.getvalue()


def make_picking_excel(result: PickingResult) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="拣货人效排名", index=False)
        result.file_summary.to_excel(writer, sheet_name="上传文件汇总", index=False)
        method = pd.DataFrame(
            {
                "项目": [
                    "R区",
                    "多文件合并",
                    "拣货件量",
                    "订单量",
                    "任务单量",
                    "任务件比",
                    "实际拣货时长",
                    "人效（小时单量）",
                    "小时任务单量",
                    "人员范围",
                ],
                "计算口径": [
                    "所属储区等于R的记录全部排除，本阶段不参与计算",
                    "允许同时上传多个周文件；完全重复的明细仅保留一条",
                    "实际拣货量合计",
                    "订单号去重数量",
                    "任务单号去重数量",
                    "拣货件量 ÷ 任务单量",
                    "每个任务取任务领取时间至最晚拣货完成时间；同一人员同一天的任务区间去除重叠后相加",
                    "订单量 ÷ 实际拣货小时（订单/小时，越高越好）",
                    "任务单量 ÷ 实际拣货小时（任务/小时，越高越好）",
                    "仅保留有任务领取时间且成功匹配人员主数据的员工",
                ],
            }
        )
        method.to_excel(writer, sheet_name="计算口径", index=False)

        workbook = writer.book
        fmt = _excel_formats(workbook)
        sheet = writer.sheets["拣货人效排名"]
        sheet.freeze_panes(1, 0)
        sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        sheet.set_row(0, 24, fmt["header"])
        widths = [8, 22, 22, 14, 12, 12, 12, 18, 18, 16]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(0, 0, 8, fmt["integer"])
        sheet.set_column(3, 5, 14, fmt["integer"])
        sheet.set_column(6, 6, 12, fmt["decimal"])
        sheet.set_column(8, 9, 18, fmt["decimal"])

        file_sheet = writer.sheets["上传文件汇总"]
        file_sheet.set_row(0, 24, fmt["header"])
        file_sheet.set_column("A:A", 45)
        file_sheet.set_column("B:B", 14, fmt["integer"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 105)
    return output.getvalue()


def render_employee_upload() -> EmployeeLookup | None:
    employee_file = st.sidebar.file_uploader(
        "1. 人员主数据",
        type=["xlsx", "xls"],
        help="验收与拣货共用。支持用户编码/Use ID、ERP和姓名字段。",
        key="employee_master",
    )
    if employee_file is None:
        return None

    try:
        file_bytes = employee_file.getvalue()
        sheets = get_sheet_names(file_bytes, employee_file.name)
        sheet_name = st.sidebar.selectbox("人员表工作表", sheets, key="employee_sheet")
        employee_df = read_excel_sheet(file_bytes, employee_file.name, sheet_name)
        lookup = build_employee_lookup(employee_df)
        st.sidebar.success(f"人员主数据已读取：{lookup.source_rows:,} 行")
        if lookup.duplicate_keys:
            st.sidebar.warning(
                f"发现 {len(lookup.duplicate_keys)} 个重复匹配键，保留首次出现的姓名。"
            )
        return lookup
    except Exception as exc:
        st.sidebar.error(f"人员表读取失败：{exc}")
        return None


def render_acceptance_module(employee_lookup: EmployeeLookup | None) -> None:
    acceptance_file = st.sidebar.file_uploader(
        "2. 验收明细表",
        type=["xlsx", "xls"],
        key="acceptance_data",
    )
    if acceptance_file is None:
        st.info("请上传人员主数据和验收明细表。")
        return
    if employee_lookup is None:
        st.info("请先上传人员主数据。未匹配员工不会参与结果。")
        return

    try:
        file_bytes = acceptance_file.getvalue()
        sheets = get_sheet_names(file_bytes, acceptance_file.name)
        sheet_name = st.sidebar.selectbox("验收表工作表", sheets, key="acceptance_sheet")
        with st.spinner("正在读取并计算验收人效……"):
            df = read_excel_sheet(
                file_bytes,
                acceptance_file.name,
                sheet_name,
                tuple(ACCEPTANCE_COLUMNS.values()),
            )
            result = calculate_acceptance_productivity(df, employee_lookup)
    except Exception as exc:
        st.error(f"验收数据处理失败：{exc}")
        return

    row1 = st.columns(4)
    row1[0].metric("总验收件量", f"{result.total_pieces:,.0f}")
    row1[1].metric("总验收单量", f"{result.total_orders:,}")
    row1[2].metric("整体单件比", f"{result.overall_piece_ratio:,.2f}")
    row1[3].metric("验收人数", f"{result.operator_count:,}")

    row2 = st.columns(3)
    row2[0].metric("系统操作时长", format_duration(result.total_system_seconds))
    row2[1].metric("实际验收时长", format_duration(result.total_actual_seconds))
    row2[2].metric("整体人效（小时/单）", f"{result.overall_hours_per_order:.4f}")

    st.subheader("验收人员排名")
    st.dataframe(
        result.ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "排名": st.column_config.NumberColumn(format="%d"),
            "验收件量": st.column_config.NumberColumn(format="%d"),
            "验收单量": st.column_config.NumberColumn(format="%d"),
            "单件比": st.column_config.NumberColumn(format="%.2f"),
            "人效（小时单量）": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.caption(
        f"有效且已匹配记录：{result.valid_row_count:,}/{result.total_row_count:,} 行；"
        f"无效记录 {result.invalid_row_count:,} 行；"
        f"未匹配人员记录 {result.excluded_unmatched_rows:,} 行未呈现。"
    )

    st.download_button(
        "下载验收人效结果 Excel",
        data=make_acceptance_excel(result),
        file_name="验收人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("验收计算口径"):
        st.markdown(
            """
- **验收件量**：验收量合计。
- **验收单量**：京东入库单号去重数量。
- **单件比**：验收件量 ÷ 验收单量。
- **系统操作时长**：原始操作区间按人员、按自然日去除重叠后相加。
- **实际验收时长**：每个PI取最早开始至最晚结束，再按人员、按自然日去除PI之间的重叠后相加。
- **人效（小时单量）**：实际验收小时 ÷ 验收单量，单位为小时/单，越低表示平均每单耗时越短。
- 未匹配人员不呈现，也不进入总数。
"""
        )


def render_picking_module(employee_lookup: EmployeeLookup | None) -> None:
    picking_files = st.sidebar.file_uploader(
        "2. 拣货结果文件（可多选）",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="可同时上传多周文件。程序会自动合并，并去除完全重复的明细。",
        key="picking_data",
    )
    if not picking_files:
        st.info("请上传人员主数据和一个或多个拣货结果文件。")
        return
    if employee_lookup is None:
        st.info("请先上传人员主数据。未匹配员工不会参与结果。")
        return

    try:
        with st.spinner(f"正在合并并计算 {len(picking_files)} 个拣货文件……"):
            combined, file_summary = combine_picking_files(picking_files)
            result = calculate_picking_productivity(
                combined,
                file_summary,
                employee_lookup,
            )
    except Exception as exc:
        st.error(f"拣货数据处理失败：{exc}")
        return

    row1 = st.columns(4)
    row1[0].metric("总拣货件量", f"{result.total_pieces:,.0f}")
    row1[1].metric("总订单量", f"{result.total_orders:,}")
    row1[2].metric("总任务单量", f"{result.total_tasks:,}")
    row1[3].metric("拣货人数", f"{result.operator_count:,}")

    row2 = st.columns(4)
    row2[0].metric("总实际拣货时长", format_duration(result.total_actual_seconds))
    row2[1].metric("整体任务件比", f"{result.overall_task_piece_ratio:,.2f}")
    row2[2].metric("整体小时单量", f"{result.overall_hourly_orders:,.2f}")
    row2[3].metric("整体小时任务单量", f"{result.overall_hourly_tasks:,.2f}")

    st.subheader("拣货人员排名")
    st.dataframe(
        result.ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "排名": st.column_config.NumberColumn(format="%d"),
            "拣货件量": st.column_config.NumberColumn(format="%d"),
            "订单量": st.column_config.NumberColumn(format="%d"),
            "任务单量": st.column_config.NumberColumn(format="%d"),
            "任务件比": st.column_config.NumberColumn(format="%.2f"),
            "人效（小时单量）": st.column_config.NumberColumn(format="%.2f"),
            "小时任务单量": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    with st.expander("查看上传文件汇总"):
        st.dataframe(result.file_summary, use_container_width=True, hide_index=True)

    duplicate_removed = result.uploaded_row_count - result.deduplicated_row_count
    st.caption(
        f"上传原始记录 {result.uploaded_row_count:,} 行；"
        f"跨文件重复明细去除 {duplicate_removed:,} 行；"
        f"R区排除 {result.r_area_row_count:,} 行；"
        f"无任务领取/完成时间 {result.missing_time_row_count:,} 行；"
        f"未匹配人员记录 {result.excluded_unmatched_rows:,} 行未呈现；"
        f"最终有效记录 {result.valid_row_count:,} 行。"
    )

    st.download_button(
        "下载拣货人效结果 Excel",
        data=make_picking_excel(result),
        file_name="拣货人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("拣货计算口径"):
        st.markdown(
            """
- 可一次上传多个周文件；程序先合并，再删除完全重复的明细。
- **R区不参与本阶段计算**：所属储区等于 `R` 的记录全部排除。
- 没有任务领取时间或拣货完成时间的记录不参与人效。
- **拣货件量**：实际拣货量合计。
- **订单量**：订单号去重数量。
- **任务单量**：任务单号去重数量。
- **任务件比**：拣货件量 ÷ 任务单量。
- **实际拣货时长**：每个任务从任务领取时间到该任务最晚拣货完成时间；同一人员同一天内的任务时间发生重叠时，重叠部分只计算一次。
- **人效（小时单量）**：订单量 ÷ 实际拣货小时，单位为订单/小时，越高越好。
- **小时任务单量**：任务单量 ÷ 实际拣货小时，单位为任务/小时，越高越好。
- 未匹配人员不呈现，也不进入总数。
"""
        )


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.caption("统一人员主数据｜验收与拣货人效分析｜结果可下载为Excel")

    with st.sidebar:
        st.header("业务模块")
        module = st.radio("选择分析环节", ["验收", "拣货"], horizontal=True)
        st.divider()
        st.header("数据上传")

    employee_lookup = render_employee_upload()

    if module == "验收":
        render_acceptance_module(employee_lookup)
    else:
        render_picking_module(employee_lookup)


if __name__ == "__main__":
    render_app()
