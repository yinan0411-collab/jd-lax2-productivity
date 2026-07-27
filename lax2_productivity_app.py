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
    "attendance_group": ["考勤组", "考勤组名称", "Attendance Group", "AttendanceGroup"],
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

PACKING_COLUMNS = {
    "time": "打包时间",
    "order": "订单号",
    "package": "包裹编号",
    "product": "商品编码",
    "quantity": "实际打包件数",
    "station": "打包台号",
    "account": "打包人账号",
    "employee": "打包人姓名",
}

PUTAWAY_COLUMNS = {
    "pi": "京东入库单号",
    "operation_type": "作业类型",
    "container": "上架容器号",
    "quantity": "上架量",
    "area_name": "储区名称",
    "operator": "上架员",
    "time": "上架时间",
}

ROBOT_PICKING_COLUMN_ALIASES = {
    "task": ["任务单号"],
    "move_task": ["搬运任务", "搬运任务号"],
    "slot": ["格口号", "格口"],
    "quantity": ["实际数量", "实际拣货量"],
    "operator": ["更新人", "操作人"],
    "time": ["更新时间", "操作时间"],
}
ROBOT_PICKING_COLUMNS = {
    "task": "任务单号",
    "move_task": "搬运任务",
    "slot": "格口号",
    "quantity": "实际数量",
    "operator": "更新人",
    "time": "更新时间",
}

PICKING_RANK_GROUP_ORDER = ["1st Shift", "2nd Shift", "其他组"]
PUTAWAY_RANK_GROUP_ORDER = ["入库-上架组IB-Putaway", "其他组"]
PACKING_RANK_GROUP_ORDER = [
    "出库-打包-Babylist",
    "出库-打包-Mix",
    "出库-Trafilea B2B",
    "其他组",
]


@dataclass(frozen=True)
class EmployeeLookup:
    exact_user_id: dict[str, str]
    exact_erp: dict[str, str]
    normalized_user_id: dict[str, str]
    normalized_erp: dict[str, str]
    exact_user_id_group: dict[str, str]
    exact_erp_group: dict[str, str]
    normalized_user_id_group: dict[str, str]
    normalized_erp_group: dict[str, str]
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
    overall_hourly_pieces: float
    operator_count: int
    uploaded_row_count: int
    deduplicated_row_count: int
    valid_row_count: int
    r_area_row_count: int
    missing_time_row_count: int
    excluded_unmatched_rows: int


@dataclass(frozen=True)
class PackingResult:
    ranking: pd.DataFrame
    file_summary: pd.DataFrame
    total_pieces: float
    total_orders: int
    total_effective_seconds: float
    overall_piece_order_ratio: float
    overall_hourly_orders: float
    overall_hourly_pieces: float
    operator_count: int
    uploaded_row_count: int
    deduplicated_row_count: int
    valid_row_count: int
    invalid_row_count: int
    excluded_unmatched_rows: int


@dataclass(frozen=True)
class PutawayResult:
    ranking: pd.DataFrame
    file_summary: pd.DataFrame
    total_pieces: float
    total_pis: int
    total_containers: int
    total_effective_seconds: float
    overall_container_piece_ratio: float
    overall_pi_container_ratio: float
    overall_hourly_pis: float
    overall_hourly_containers: float
    overall_hourly_pieces: float
    operator_count: int
    uploaded_row_count: int
    deduplicated_row_count: int
    valid_row_count: int
    robot_row_count: int
    invalid_row_count: int
    excluded_unmatched_rows: int
    operation_type: str


@dataclass(frozen=True)
class RobotPickingResult:
    ranking: pd.DataFrame
    file_summary: pd.DataFrame
    total_pieces: float
    total_tasks: int
    total_slots: int
    total_effective_seconds: float
    overall_piece_task_ratio: float
    overall_task_slot_ratio: float
    overall_slot_piece_ratio: float
    overall_hourly_tasks: float
    overall_hourly_slots: float
    overall_hourly_pieces: float
    operator_count: int
    uploaded_row_count: int
    deduplicated_row_count: int
    valid_row_count: int
    invalid_row_count: int
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


def classify_picking_rank_group(attendance_group: object) -> str:
    """Collapse picking attendance groups into 1st Shift, 2nd Shift, or 其他组."""
    text = canonical_key(attendance_group)
    if "1st shift" in text:
        return "1st Shift"
    if "2nd shift" in text:
        return "2nd Shift"
    return "其他组"


def classify_packing_rank_group(attendance_group: object) -> str:
    """Keep the three designated packing groups; place every other value in 其他组."""
    text = canonical_key(attendance_group)
    target_map = {canonical_key(group): group for group in PACKING_RANK_GROUP_ORDER[:-1]}
    return target_map.get(text, "其他组")


def classify_putaway_rank_group(attendance_group: object) -> str:
    """Keep IB Putaway in its own ranking group; place all others in 其他组."""
    text = canonical_key(attendance_group)
    target = PUTAWAY_RANK_GROUP_ORDER[0]
    return target if canonical_key(target) in text else "其他组"


def apply_group_ranking(
    summary: pd.DataFrame,
    group_col: str,
    group_order: Sequence[str],
    sort_columns: Sequence[str],
    ascending: Sequence[bool],
) -> pd.DataFrame:
    """Sort employees by configured group order and reset ranking within each group."""
    ranked = summary.copy()
    ranked["_排名组顺序"] = pd.Categorical(
        ranked[group_col], categories=list(group_order), ordered=True
    )
    ranked = ranked.sort_values(
        ["_排名组顺序", *sort_columns],
        ascending=[True, *ascending],
        na_position="last",
    ).reset_index(drop=True)
    ranked["组内排名"] = ranked.groupby(group_col, sort=False).cumcount() + 1
    return ranked.drop(columns=["_排名组顺序"])


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


def choose_packing_operator_id(employee_value: object, account_value: object) -> str:
    """Use the ID-like value in 打包人姓名 first, then fall back to 打包人账号."""
    employee = extract_operator_id(employee_value)
    if employee:
        return employee
    return extract_operator_id(account_value)


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
    attendance_group_col = find_column(
        employee_df.columns, EMPLOYEE_COLUMN_ALIASES["attendance_group"]
    )

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
    exact_user_id_group: dict[str, str] = {}
    exact_erp_group: dict[str, str] = {}
    normalized_user_id_group: dict[str, str] = {}
    normalized_erp_group: dict[str, str] = {}
    duplicate_keys: list[str] = []

    for _, row in employee_df.iterrows():
        name = clean_name(row.get(name_col))
        if not name:
            continue
        attendance_group = (
            clean_name(row.get(attendance_group_col)) if attendance_group_col else ""
        )

        if user_id_col is not None:
            raw_user_id = row.get(user_id_col)
            exact_key = canonical_key(raw_user_id)
            normalized_key = compact_key(raw_user_id)
            _add_lookup_value(
                exact_user_id,
                exact_key,
                name,
                duplicate_keys,
                "用户编码",
            )
            _add_lookup_value(
                normalized_user_id,
                normalized_key,
                name,
                duplicate_keys,
                "标准化用户编码",
            )
            if exact_key:
                exact_user_id_group.setdefault(exact_key, attendance_group)
            if normalized_key:
                normalized_user_id_group.setdefault(normalized_key, attendance_group)

        if erp_col is not None:
            raw_erp = row.get(erp_col)
            exact_key = canonical_key(raw_erp)
            normalized_key = compact_key(raw_erp)
            _add_lookup_value(
                exact_erp,
                exact_key,
                name,
                duplicate_keys,
                "ERP",
            )
            _add_lookup_value(
                normalized_erp,
                normalized_key,
                name,
                duplicate_keys,
                "标准化ERP",
            )
            if exact_key:
                exact_erp_group.setdefault(exact_key, attendance_group)
            if normalized_key:
                normalized_erp_group.setdefault(normalized_key, attendance_group)

    return EmployeeLookup(
        exact_user_id=exact_user_id,
        exact_erp=exact_erp,
        normalized_user_id=normalized_user_id,
        normalized_erp=normalized_erp,
        exact_user_id_group=exact_user_id_group,
        exact_erp_group=exact_erp_group,
        normalized_user_id_group=normalized_user_id_group,
        normalized_erp_group=normalized_erp_group,
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


def match_employee_attendance_group(
    operator_id: str, lookup: EmployeeLookup | None
) -> str:
    """Return the employee attendance group; missing source values remain blank."""
    if lookup is None:
        return ""

    exact = canonical_key(operator_id)
    compact = compact_key(operator_id)

    if exact in lookup.exact_user_id:
        return lookup.exact_user_id_group.get(exact, "")
    if exact in lookup.exact_erp:
        return lookup.exact_erp_group.get(exact, "")
    if compact and compact in lookup.normalized_user_id:
        return lookup.normalized_user_id_group.get(compact, "")
    if compact and compact in lookup.normalized_erp:
        return lookup.normalized_erp_group.get(compact, "")
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
    work["考勤组"] = work["验收人账号"].map(
        lambda value: match_employee_attendance_group(value, employee_lookup)
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
        valid.groupby(["验收人账号", "实际姓名", "考勤组"], as_index=False)
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
        ["验收单量", "验收件量", "人效（小时单量）"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    summary.insert(0, "排名", np.arange(1, len(summary) + 1))

    ranking = summary[
        [
            "排名",
            "验收人账号",
            "实际姓名",
            "考勤组",
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
    work["考勤组"] = work["拣货人账号"].map(
        lambda value: match_employee_attendance_group(value, employee_lookup)
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
        valid.groupby(["拣货人账号", "实际姓名", "考勤组"], as_index=False)
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
    summary["人效（小时件效）"] = np.where(
        summary["实际拣货小时"] > 0,
        summary["拣货件量"] / summary["实际拣货小时"],
        np.nan,
    )
    summary["实际拣货时长"] = summary["实际拣货秒数"].map(format_duration)
    summary["排名组"] = summary["考勤组"].map(classify_picking_rank_group)

    summary = apply_group_ranking(
        summary,
        group_col="排名组",
        group_order=PICKING_RANK_GROUP_ORDER,
        sort_columns=["订单量", "拣货件量", "人效（小时单量）"],
        ascending=[False, False, False],
    )

    ranking = summary[
        [
            "排名组",
            "组内排名",
            "拣货人账号",
            "实际姓名",
            "考勤组",
            "拣货件量",
            "订单量",
            "任务单量",
            "任务件比",
            "实际拣货时长",
            "人效（小时单量）",
            "小时任务单量",
            "人效（小时件效）",
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
        overall_hourly_pieces=(total_pieces / total_hours if total_hours else float("nan")),
        operator_count=len(summary),
        uploaded_row_count=uploaded_row_count,
        deduplicated_row_count=deduplicated_row_count,
        valid_row_count=len(valid),
        r_area_row_count=r_area_row_count,
        missing_time_row_count=missing_time_row_count,
        excluded_unmatched_rows=excluded_unmatched_rows,
    )



def combine_packing_files(uploaded_files: Sequence[object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    required = tuple(PACKING_COLUMNS.values())

    for file_index, uploaded in enumerate(uploaded_files):
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        try:
            frame = read_excel_sheet(file_bytes, file_name, 0, required)
        except ValueError as exc:
            raise ValueError(f"文件“{file_name}”缺少必要字段或读取失败：{exc}") from exc

        original_rows = len(frame)
        # Preserve identical rows within one source file. The occurrence number
        # removes only repeated copies caused by overlapping uploaded files.
        frame["_文件内重复序号"] = frame.groupby(
            list(required), sort=False, dropna=False
        ).cumcount()
        frame["_来源文件序号"] = file_index
        frame["来源文件"] = file_name
        frames.append(frame)
        summary_rows.append({"文件名": file_name, "读取行数": original_rows})

    if not frames:
        raise ValueError("请至少上传一个打包文件。")

    combined = pd.concat(frames, ignore_index=True)
    file_summary = pd.DataFrame(summary_rows)
    return combined, file_summary


def combine_putaway_files(uploaded_files: Sequence[object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    required = tuple(PUTAWAY_COLUMNS.values())

    for file_index, uploaded in enumerate(uploaded_files):
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        try:
            frame = read_excel_sheet(file_bytes, file_name, 0, required)
        except ValueError as exc:
            raise ValueError(f"文件“{file_name}”缺少必要字段或读取失败：{exc}") from exc

        original_rows = len(frame)
        # Keep legitimate identical rows within one source file. The occurrence
        # number removes only repeated copies caused by overlapping uploads.
        frame["_文件内重复序号"] = frame.groupby(
            list(required), sort=False, dropna=False
        ).cumcount()
        frame["_来源文件序号"] = file_index
        frame["来源文件"] = file_name
        frames.append(frame)
        summary_rows.append({"文件名": file_name, "读取行数": original_rows})

    if not frames:
        raise ValueError("请至少上传一个上架文件。")

    combined = pd.concat(frames, ignore_index=True)
    file_summary = pd.DataFrame(summary_rows)
    return combined, file_summary


def calculate_putaway_productivity(
    combined_df: pd.DataFrame,
    file_summary: pd.DataFrame,
    employee_lookup: EmployeeLookup | None,
    gap_minutes: int,
    operation_type: str = "全部",
) -> PutawayResult:
    required = list(PUTAWAY_COLUMNS.values())
    missing = [column for column in required if column not in combined_df.columns]
    if missing:
        raise ValueError("上架表缺少必要字段：" + "、".join(missing))

    uploaded_row_count = len(combined_df)

    dedupe_key = required.copy()
    if "_文件内重复序号" in combined_df.columns:
        dedupe_key.append("_文件内重复序号")
    work = combined_df.drop_duplicates(subset=dedupe_key, keep="first").copy()
    deduplicated_row_count = len(work)

    work["作业类型标准值"] = work[PUTAWAY_COLUMNS["operation_type"]].map(clean_name)
    robot_mask = (
        work[PUTAWAY_COLUMNS["area_name"]].map(canonical_key).eq("robot")
        | work[PUTAWAY_COLUMNS["operator"]].map(canonical_key).eq("tmc_call")
    )
    robot_row_count = int(robot_mask.sum())
    work = work.loc[~robot_mask].copy()

    if operation_type and operation_type != "全部":
        work = work.loc[work["作业类型标准值"].eq(operation_type)].copy()

    work["上架人账号"] = work[PUTAWAY_COLUMNS["operator"]].map(extract_operator_id)
    work["实际姓名"] = work["上架人账号"].map(
        lambda value: match_employee_name(value, employee_lookup)
    )
    work["考勤组"] = work["上架人账号"].map(
        lambda value: match_employee_attendance_group(value, employee_lookup)
    )
    work["PI标准值"] = work[PUTAWAY_COLUMNS["pi"]].map(clean_identifier)
    work["容器号标准值"] = work[PUTAWAY_COLUMNS["container"]].map(clean_identifier)
    work["上架件量数值"] = pd.to_numeric(
        work[PUTAWAY_COLUMNS["quantity"]], errors="coerce"
    )
    work["上架完成时间"] = pd.to_datetime(
        work[PUTAWAY_COLUMNS["time"]], errors="coerce"
    )

    valid_mask = (
        work["上架人账号"].ne("")
        & work["PI标准值"].ne("")
        & work["容器号标准值"].ne("")
        & work["上架件量数值"].notna()
        & work["上架完成时间"].notna()
        & (work["上架件量数值"] >= 0)
    )
    invalid_row_count = int((~valid_mask).sum())
    work = work.loc[valid_mask].copy()

    excluded_unmatched_rows = int(work["实际姓名"].eq("").sum())
    valid = work.loc[work["实际姓名"].ne("")].copy()
    if valid.empty:
        selected_label = f"（作业类型：{operation_type}）" if operation_type != "全部" else ""
        raise ValueError(
            f"排除Robot、无效数据和未匹配人员后{selected_label}，没有可计算的上架记录。"
        )

    # Each valid source row is counted as one completed container operation.
    # The container number itself is intentionally NOT deduplicated.
    duration = calculate_gap_based_seconds(
        valid,
        "上架人账号",
        "上架完成时间",
        gap_minutes,
    ).rename(columns={"账号": "上架人账号", "秒数": "有效上架秒数"})

    summary = (
        valid.groupby(["上架人账号", "实际姓名", "考勤组"], as_index=False)
        .agg(
            上架件量=("上架件量数值", "sum"),
            上架单量=("PI标准值", "nunique"),
            上架容器量=("容器号标准值", "count"),
        )
    )
    summary = summary.merge(duration, on="上架人账号", how="left")
    summary["有效上架秒数"] = summary["有效上架秒数"].fillna(0.0)
    summary["有效上架小时"] = summary["有效上架秒数"] / 3600
    summary["容器件比"] = np.where(
        summary["上架容器量"] > 0,
        summary["上架件量"] / summary["上架容器量"],
        np.nan,
    )
    summary["PI容器比"] = np.where(
        summary["上架单量"] > 0,
        summary["上架容器量"] / summary["上架单量"],
        np.nan,
    )
    summary["人效（小时单量）"] = np.where(
        summary["有效上架小时"] > 0,
        summary["上架单量"] / summary["有效上架小时"],
        np.nan,
    )
    summary["小时上架容器量"] = np.where(
        summary["有效上架小时"] > 0,
        summary["上架容器量"] / summary["有效上架小时"],
        np.nan,
    )
    summary["人效（小时件效）"] = np.where(
        summary["有效上架小时"] > 0,
        summary["上架件量"] / summary["有效上架小时"],
        np.nan,
    )
    summary["有效上架时长"] = summary["有效上架秒数"].map(format_duration)
    summary["排名组"] = summary["考勤组"].map(classify_putaway_rank_group)

    summary = apply_group_ranking(
        summary,
        group_col="排名组",
        group_order=PUTAWAY_RANK_GROUP_ORDER,
        sort_columns=["上架容器量", "上架件量", "小时上架容器量"],
        ascending=[False, False, False],
    )

    ranking = summary[
        [
            "排名组",
            "组内排名",
            "上架人账号",
            "实际姓名",
            "考勤组",
            "上架件量",
            "上架单量",
            "上架容器量",
            "容器件比",
            "PI容器比",
            "有效上架时长",
            "人效（小时单量）",
            "小时上架容器量",
            "人效（小时件效）",
        ]
    ].copy()

    total_pieces = float(valid["上架件量数值"].sum())
    total_pis = int(valid["PI标准值"].nunique())
    total_containers = int(valid["容器号标准值"].count())
    total_effective_seconds = float(summary["有效上架秒数"].sum())
    total_hours = total_effective_seconds / 3600

    return PutawayResult(
        ranking=ranking,
        file_summary=file_summary,
        total_pieces=total_pieces,
        total_pis=total_pis,
        total_containers=total_containers,
        total_effective_seconds=total_effective_seconds,
        overall_container_piece_ratio=(
            total_pieces / total_containers if total_containers else float("nan")
        ),
        overall_pi_container_ratio=(
            total_containers / total_pis if total_pis else float("nan")
        ),
        overall_hourly_pis=(total_pis / total_hours if total_hours else float("nan")),
        overall_hourly_containers=(
            total_containers / total_hours if total_hours else float("nan")
        ),
        overall_hourly_pieces=(
            total_pieces / total_hours if total_hours else float("nan")
        ),
        operator_count=len(summary),
        uploaded_row_count=uploaded_row_count,
        deduplicated_row_count=deduplicated_row_count,
        valid_row_count=len(valid),
        robot_row_count=robot_row_count,
        invalid_row_count=invalid_row_count,
        excluded_unmatched_rows=excluded_unmatched_rows,
        operation_type=operation_type,
    )


def combine_robot_picking_files(
    uploaded_files: Sequence[object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    canonical_columns = list(ROBOT_PICKING_COLUMNS.values())

    for file_index, uploaded in enumerate(uploaded_files):
        file_bytes = uploaded.getvalue()
        file_name = uploaded.name
        try:
            raw = read_excel_sheet(file_bytes, file_name, 0, None)
        except Exception as exc:
            raise ValueError(f"文件“{file_name}”读取失败：{exc}") from exc

        actual_columns: dict[str, str] = {}
        missing_labels: list[str] = []
        for key, aliases in ROBOT_PICKING_COLUMN_ALIASES.items():
            actual = find_column(raw.columns, aliases)
            if actual is None:
                missing_labels.append("/".join(aliases))
            else:
                actual_columns[key] = actual
        if missing_labels:
            raise ValueError(
                f"文件“{file_name}”缺少必要字段：" + "、".join(missing_labels)
            )

        frame = raw[[actual_columns[key] for key in ROBOT_PICKING_COLUMNS]].copy()
        frame.columns = canonical_columns
        original_rows = len(frame)
        frame["_文件内重复序号"] = frame.groupby(
            canonical_columns, sort=False, dropna=False
        ).cumcount()
        frame["_来源文件序号"] = file_index
        frame["来源文件"] = file_name
        frames.append(frame)
        summary_rows.append({"文件名": file_name, "读取行数": original_rows})

    if not frames:
        raise ValueError("请至少上传一个机区拣货文件。")

    combined = pd.concat(frames, ignore_index=True)
    return combined, pd.DataFrame(summary_rows)


def calculate_robot_picking_productivity(
    combined_df: pd.DataFrame,
    file_summary: pd.DataFrame,
    employee_lookup: EmployeeLookup | None,
    gap_minutes: int,
) -> RobotPickingResult:
    required = list(ROBOT_PICKING_COLUMNS.values())
    missing = [column for column in required if column not in combined_df.columns]
    if missing:
        raise ValueError("机区拣货表缺少必要字段：" + "、".join(missing))

    uploaded_row_count = len(combined_df)
    dedupe_key = required.copy()
    if "_文件内重复序号" in combined_df.columns:
        dedupe_key.append("_文件内重复序号")
    work = combined_df.drop_duplicates(subset=dedupe_key, keep="first").copy()
    deduplicated_row_count = len(work)

    work["更新人账号"] = work[ROBOT_PICKING_COLUMNS["operator"]].map(extract_operator_id)
    work["实际姓名"] = work["更新人账号"].map(
        lambda value: match_employee_name(value, employee_lookup)
    )
    work["考勤组"] = work["更新人账号"].map(
        lambda value: match_employee_attendance_group(value, employee_lookup)
    )
    work["任务单号标准值"] = work[ROBOT_PICKING_COLUMNS["task"]].map(clean_identifier)
    work["格口号标准值"] = work[ROBOT_PICKING_COLUMNS["slot"]].map(clean_identifier)
    work["机区拣货件量数值"] = pd.to_numeric(
        work[ROBOT_PICKING_COLUMNS["quantity"]], errors="coerce"
    )
    work["机区操作时间"] = pd.to_datetime(
        work[ROBOT_PICKING_COLUMNS["time"]], errors="coerce"
    )

    valid_mask = (
        work["更新人账号"].ne("")
        & work["任务单号标准值"].ne("")
        & work["格口号标准值"].ne("")
        & work["机区拣货件量数值"].notna()
        & work["机区操作时间"].notna()
        & (work["机区拣货件量数值"] >= 0)
    )
    invalid_row_count = int((~valid_mask).sum())
    work = work.loc[valid_mask].copy()

    excluded_unmatched_rows = int(work["实际姓名"].eq("").sum())
    valid = work.loc[work["实际姓名"].ne("")].copy()
    if valid.empty:
        raise ValueError("排除无效数据和未匹配人员后，没有可计算的机区拣货记录。")

    duration = calculate_gap_based_seconds(
        valid,
        "更新人账号",
        "机区操作时间",
        gap_minutes,
    ).rename(columns={"账号": "更新人账号", "秒数": "有效机区拣货秒数"})

    summary = (
        valid.groupby(["更新人账号", "实际姓名", "考勤组"], as_index=False)
        .agg(
            机区拣货件量=("机区拣货件量数值", "sum"),
            机区任务单量=("任务单号标准值", "nunique"),
            机区格口量=("格口号标准值", "count"),
        )
    )
    summary = summary.merge(duration, on="更新人账号", how="left")
    summary["有效机区拣货秒数"] = summary["有效机区拣货秒数"].fillna(0.0)
    summary["有效机区拣货小时"] = summary["有效机区拣货秒数"] / 3600
    summary["单件比"] = np.where(
        summary["机区任务单量"] > 0,
        summary["机区拣货件量"] / summary["机区任务单量"],
        np.nan,
    )
    summary["任务格口比"] = np.where(
        summary["机区任务单量"] > 0,
        summary["机区格口量"] / summary["机区任务单量"],
        np.nan,
    )
    summary["格口件比"] = np.where(
        summary["机区格口量"] > 0,
        summary["机区拣货件量"] / summary["机区格口量"],
        np.nan,
    )
    summary["人效（小时单量）"] = np.where(
        summary["有效机区拣货小时"] > 0,
        summary["机区任务单量"] / summary["有效机区拣货小时"],
        np.nan,
    )
    summary["小时机区格口量"] = np.where(
        summary["有效机区拣货小时"] > 0,
        summary["机区格口量"] / summary["有效机区拣货小时"],
        np.nan,
    )
    summary["人效（小时件效）"] = np.where(
        summary["有效机区拣货小时"] > 0,
        summary["机区拣货件量"] / summary["有效机区拣货小时"],
        np.nan,
    )
    summary["有效机区拣货时长"] = summary["有效机区拣货秒数"].map(format_duration)
    summary["排名组"] = summary["考勤组"].map(classify_picking_rank_group)

    summary = apply_group_ranking(
        summary,
        group_col="排名组",
        group_order=PICKING_RANK_GROUP_ORDER,
        sort_columns=["机区任务单量", "机区格口量", "机区拣货件量", "人效（小时单量）"],
        ascending=[False, False, False, False],
    )

    ranking = summary[
        [
            "排名组",
            "组内排名",
            "更新人账号",
            "实际姓名",
            "考勤组",
            "机区拣货件量",
            "机区任务单量",
            "机区格口量",
            "单件比",
            "任务格口比",
            "格口件比",
            "有效机区拣货时长",
            "人效（小时单量）",
            "小时机区格口量",
            "人效（小时件效）",
        ]
    ].copy()

    total_pieces = float(valid["机区拣货件量数值"].sum())
    total_tasks = int(valid["任务单号标准值"].nunique())
    total_slots = int(valid["格口号标准值"].count())
    total_effective_seconds = float(summary["有效机区拣货秒数"].sum())
    total_hours = total_effective_seconds / 3600

    return RobotPickingResult(
        ranking=ranking,
        file_summary=file_summary,
        total_pieces=total_pieces,
        total_tasks=total_tasks,
        total_slots=total_slots,
        total_effective_seconds=total_effective_seconds,
        overall_piece_task_ratio=(
            total_pieces / total_tasks if total_tasks else float("nan")
        ),
        overall_task_slot_ratio=(
            total_slots / total_tasks if total_tasks else float("nan")
        ),
        overall_slot_piece_ratio=(
            total_pieces / total_slots if total_slots else float("nan")
        ),
        overall_hourly_tasks=(
            total_tasks / total_hours if total_hours else float("nan")
        ),
        overall_hourly_slots=(
            total_slots / total_hours if total_hours else float("nan")
        ),
        overall_hourly_pieces=(
            total_pieces / total_hours if total_hours else float("nan")
        ),
        operator_count=len(summary),
        uploaded_row_count=uploaded_row_count,
        deduplicated_row_count=deduplicated_row_count,
        valid_row_count=len(valid),
        invalid_row_count=invalid_row_count,
        excluded_unmatched_rows=excluded_unmatched_rows,
    )


def calculate_gap_based_seconds(
    order_events: pd.DataFrame,
    operator_col: str,
    time_col: str,
    gap_minutes: int,
) -> pd.DataFrame:
    """Estimate active time from adjacent completed-order timestamps by person and day."""
    max_gap = pd.Timedelta(minutes=gap_minutes)
    work = order_events[[operator_col, time_col]].dropna().copy()
    work["工作日期"] = work[time_col].dt.date

    daily_records: list[dict[str, object]] = []
    for (operator_id, work_date), group in work.groupby(
        [operator_col, "工作日期"], sort=False
    ):
        times = group[time_col].sort_values().drop_duplicates()
        if len(times) < 2:
            seconds = 0.0
        else:
            gaps = times.diff().dropna()
            valid_gaps = gaps[(gaps >= pd.Timedelta(0)) & (gaps <= max_gap)]
            seconds = float(valid_gaps.dt.total_seconds().sum())
        daily_records.append(
            {
                "账号": operator_id,
                "工作日期": work_date,
                "秒数": seconds,
            }
        )

    if not daily_records:
        return pd.DataFrame(columns=["账号", "秒数"])

    daily = pd.DataFrame(daily_records)
    return daily.groupby("账号", as_index=False)["秒数"].sum()


def calculate_packing_productivity(
    combined_df: pd.DataFrame,
    file_summary: pd.DataFrame,
    employee_lookup: EmployeeLookup | None,
    gap_minutes: int,
) -> PackingResult:
    required = list(PACKING_COLUMNS.values())
    missing = [column for column in required if column not in combined_df.columns]
    if missing:
        raise ValueError("打包表缺少必要字段：" + "、".join(missing))

    uploaded_row_count = len(combined_df)

    # Remove repeated copies across overlapping files while preserving legitimate
    # identical rows that already existed inside a single source file.
    dedupe_key = required.copy()
    if "_文件内重复序号" in combined_df.columns:
        dedupe_key.append("_文件内重复序号")
    work = combined_df.drop_duplicates(subset=dedupe_key, keep="first").copy()
    deduplicated_row_count = len(work)

    work["打包人账号"] = [
        choose_packing_operator_id(employee_value, account_value)
        for employee_value, account_value in zip(
            work[PACKING_COLUMNS["employee"]],
            work[PACKING_COLUMNS["account"]],
        )
    ]
    work["实际姓名"] = work["打包人账号"].map(
        lambda value: match_employee_name(value, employee_lookup)
    )
    work["考勤组"] = work["打包人账号"].map(
        lambda value: match_employee_attendance_group(value, employee_lookup)
    )
    work["订单号标准值"] = work[PACKING_COLUMNS["order"]].map(clean_identifier)
    work["打包件量数值"] = pd.to_numeric(
        work[PACKING_COLUMNS["quantity"]], errors="coerce"
    )
    work["打包完成时间"] = pd.to_datetime(
        work[PACKING_COLUMNS["time"]], errors="coerce"
    )

    valid_mask = (
        work["打包人账号"].ne("")
        & work["订单号标准值"].ne("")
        & work["打包件量数值"].notna()
        & work["打包完成时间"].notna()
        & (work["打包件量数值"] >= 0)
    )
    invalid_row_count = int((~valid_mask).sum())
    work = work.loc[valid_mask].copy()

    excluded_unmatched_rows = int(work["实际姓名"].eq("").sum())
    valid = work.loc[work["实际姓名"].ne("")].copy()
    if valid.empty:
        raise ValueError("排除无效数据和未匹配人员后，没有可计算的打包记录。")

    # One order counts once. When an order has more than one completion timestamp,
    # use its latest timestamp as the order's completion event.
    order_events = (
        valid.groupby(
            ["打包人账号", "实际姓名", "考勤组", "订单号标准值"], as_index=False
        )
        .agg(
            打包件量=("打包件量数值", "sum"),
            订单完成时间=("打包完成时间", "max"),
        )
    )

    duration = calculate_gap_based_seconds(
        order_events,
        "打包人账号",
        "订单完成时间",
        gap_minutes,
    ).rename(columns={"账号": "打包人账号", "秒数": "有效打包秒数"})

    summary = (
        order_events.groupby(["打包人账号", "实际姓名", "考勤组"], as_index=False)
        .agg(
            打包件量=("打包件量", "sum"),
            订单量=("订单号标准值", "nunique"),
        )
    )
    summary = summary.merge(duration, on="打包人账号", how="left")
    summary["有效打包秒数"] = summary["有效打包秒数"].fillna(0.0)
    summary["有效打包小时"] = summary["有效打包秒数"] / 3600
    summary["件单比"] = np.where(
        summary["订单量"] > 0,
        summary["打包件量"] / summary["订单量"],
        np.nan,
    )
    summary["人效（小时单量）"] = np.where(
        summary["有效打包小时"] > 0,
        summary["订单量"] / summary["有效打包小时"],
        np.nan,
    )
    summary["人效（小时件效）"] = np.where(
        summary["有效打包小时"] > 0,
        summary["打包件量"] / summary["有效打包小时"],
        np.nan,
    )
    summary["有效打包时长"] = summary["有效打包秒数"].map(format_duration)
    summary["排名组"] = summary["考勤组"].map(classify_packing_rank_group)

    summary = apply_group_ranking(
        summary,
        group_col="排名组",
        group_order=PACKING_RANK_GROUP_ORDER,
        sort_columns=["订单量", "打包件量", "人效（小时单量）"],
        ascending=[False, False, False],
    )

    ranking = summary[
        [
            "排名组",
            "组内排名",
            "打包人账号",
            "实际姓名",
            "考勤组",
            "打包件量",
            "订单量",
            "件单比",
            "有效打包时长",
            "人效（小时单量）",
            "人效（小时件效）",
        ]
    ].copy()

    total_pieces = float(order_events["打包件量"].sum())
    total_orders = int(order_events["订单号标准值"].nunique())
    total_effective_seconds = float(summary["有效打包秒数"].sum())
    total_hours = total_effective_seconds / 3600

    return PackingResult(
        ranking=ranking,
        file_summary=file_summary,
        total_pieces=total_pieces,
        total_orders=total_orders,
        total_effective_seconds=total_effective_seconds,
        overall_piece_order_ratio=(
            total_pieces / total_orders if total_orders else float("nan")
        ),
        overall_hourly_orders=(
            total_orders / total_hours if total_hours else float("nan")
        ),
        overall_hourly_pieces=(
            total_pieces / total_hours if total_hours else float("nan")
        ),
        operator_count=len(summary),
        uploaded_row_count=uploaded_row_count,
        deduplicated_row_count=deduplicated_row_count,
        valid_row_count=len(valid),
        invalid_row_count=invalid_row_count,
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
                    "考勤组",
                    "验收件量",
                    "验收单量",
                    "单件比",
                    "系统操作时长",
                    "实际验收时长",
                    "人效（小时单量）",
                    "排名规则",
                    "人员范围",
                ],
                "计算口径": [
                    "从人员主数据中的考勤组字段匹配；源数据为空时结果保持为空",
                    "按人员汇总验收量",
                    "京东入库单号去重数量",
                    "验收件量 ÷ 验收单量",
                    "原始系统操作区间按人员、按自然日去除重叠后相加",
                    "单个PI取最早开始至最晚结束；同一人员同一天的PI区间去除重叠后相加",
                    "实际验收小时 ÷ 验收单量（小时/单，越低表示平均每单用时越短）",
                    "优先按照实际完成的验收单量从高到低排名；单量相同时依次参考验收件量和平均每单用时",
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
        widths = [8, 22, 22, 18, 14, 14, 12, 18, 18, 18]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(0, 0, 8, fmt["integer"])
        sheet.set_column(4, 5, 14, fmt["integer"])
        sheet.set_column(6, 6, 12, fmt["decimal"])
        sheet.set_column(9, 9, 18, fmt["decimal4"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 110)
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
                    "考勤组",
                    "排名分组",
                    "拣货件量",
                    "订单量",
                    "任务单量",
                    "任务件比",
                    "实际拣货时长",
                    "人效（小时单量）",
                    "小时任务单量",
                    "人效（小时件效）",
                    "排名规则",
                    "人员范围",
                ],
                "计算口径": [
                    "所属储区等于R的记录全部排除，本阶段不参与计算",
                    "允许同时上传多个周文件；完全重复的明细仅保留一条",
                    "从人员主数据中的考勤组字段匹配；源数据为空时结果保持为空",
                    "考勤组名称包含1st Shift的归入1st Shift，包含2nd Shift的归入2nd Shift，其余全部归入其他组",
                    "实际拣货量合计",
                    "订单号去重数量",
                    "任务单号去重数量",
                    "拣货件量 ÷ 任务单量",
                    "每个任务取任务领取时间至最晚拣货完成时间；同一人员同一天的任务区间去除重叠后相加",
                    "订单量 ÷ 实际拣货小时（订单/小时，越高越好）",
                    "任务单量 ÷ 实际拣货小时（任务/小时，越高越好）",
                    "拣货件量 ÷ 实际拣货小时（件/小时，越高越好）",
                    "先按1st Shift、2nd Shift、其他组分组；各组内按照实际完成的订单量从高到低排名，订单量相同时依次参考拣货件量和小时单量",
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
        widths = [16, 10, 22, 22, 22, 14, 12, 12, 12, 18, 18, 16, 18]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(1, 1, 10, fmt["integer"])
        sheet.set_column(5, 7, 14, fmt["integer"])
        sheet.set_column(8, 8, 12, fmt["decimal"])
        sheet.set_column(10, 12, 18, fmt["decimal"])

        file_sheet = writer.sheets["上传文件汇总"]
        file_sheet.set_row(0, 24, fmt["header"])
        file_sheet.set_column("A:A", 45)
        file_sheet.set_column("B:B", 14, fmt["integer"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 110)
    return output.getvalue()

def make_packing_excel(result: PackingResult, gap_minutes: int) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="打包人效排名", index=False)
        result.file_summary.to_excel(writer, sheet_name="上传文件汇总", index=False)
        method = pd.DataFrame(
            {
                "项目": [
                    "多文件合并",
                    "考勤组",
                    "排名分组",
                    "打包件量",
                    "订单量",
                    "件单比",
                    "订单完成时间",
                    "有效打包时长",
                    "连续打包阈值",
                    "人效（小时单量）",
                    "人效（小时件效）",
                    "排名规则",
                    "人员范围",
                ],
                "计算口径": [
                    "允许同时上传多个文件；仅删除因文件时间范围重叠而重复出现的商品明细，单个源文件内的原始行保持不变",
                    "从人员主数据中的考勤组字段匹配；源数据为空时结果保持为空",
                    "仅保留出库-打包-Babylist、出库-打包-Mix、出库-Trafilea B2B三个指定组，其余考勤组全部归入其他组",
                    "实际打包件数合计",
                    "订单号去重数量；一个订单只计一次",
                    "打包件量 ÷ 订单量",
                    "同一订单出现多个打包时间时，使用最晚打包时间作为该订单的完成时间",
                    "按人员和自然日排列订单完成时间；相邻订单间隔不超过阈值时，该间隔计入有效打包时长",
                    f"当前设置为 {gap_minutes} 分钟；超过阈值的空档不计入有效打包时长",
                    "订单量 ÷ 有效打包小时（订单/小时，越高越好）",
                    "打包件量 ÷ 有效打包小时（件/小时，越高越好）",
                    "先按出库-打包-Babylist、出库-打包-Mix、出库-Trafilea B2B、其他组分组；各组内按照实际完成的订单量从高到低排名，订单量相同时依次参考打包件量和小时单量",
                    "仅保留成功匹配人员主数据的员工",
                ],
            }
        )
        method.to_excel(writer, sheet_name="计算口径", index=False)

        workbook = writer.book
        fmt = _excel_formats(workbook)
        sheet = writer.sheets["打包人效排名"]
        sheet.freeze_panes(1, 0)
        sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        sheet.set_row(0, 24, fmt["header"])
        widths = [22, 10, 22, 22, 22, 14, 12, 12, 18, 18, 18]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(1, 1, 10, fmt["integer"])
        sheet.set_column(5, 6, 14, fmt["integer"])
        sheet.set_column(7, 7, 12, fmt["decimal"])
        sheet.set_column(9, 10, 18, fmt["decimal"])

        file_sheet = writer.sheets["上传文件汇总"]
        file_sheet.set_row(0, 24, fmt["header"])
        file_sheet.set_column("A:A", 45)
        file_sheet.set_column("B:B", 14, fmt["integer"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 110)
    return output.getvalue()

def make_putaway_excel(result: PutawayResult, gap_minutes: int) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="上架人效排名", index=False)
        result.file_summary.to_excel(writer, sheet_name="上传文件汇总", index=False)
        method = pd.DataFrame(
            {
                "项目": [
                    "多文件合并",
                    "Robot排除",
                    "作业类型",
                    "考勤组",
                    "排名分组",
                    "上架件量",
                    "上架单量",
                    "上架容器量",
                    "容器件比",
                    "PI容器比",
                    "容器完成时间",
                    "有效上架时长",
                    "连续上架阈值",
                    "人效（小时单量）",
                    "小时上架容器量",
                    "人效（小时件效）",
                    "排名规则",
                    "人员范围",
                ],
                "计算口径": [
                    "允许同时上传多个文件；仅删除因文件时间范围重叠而重复出现的明细，单个源文件内的原始行保持不变",
                    "储区名称等于Robot或上架员等于TMC_CALL的记录全部排除",
                    f"当前选择：{result.operation_type}；选择全部时合并所有非Robot作业类型",
                    "从人员主数据中的考勤组字段匹配；源数据为空时结果保持为空",
                    "入库-上架组IB-Putaway单独排名，其余考勤组全部归入其他组",
                    "上架量合计",
                    "京东入库单号（PI）去重数量；一个PI可以包含多个容器",
                    "上架容器号不去重；每条有效上架记录计为1个上架容器量，同一容器号重复出现时每次均计入",
                    "上架件量 ÷ 上架容器量",
                    "上架容器量 ÷ 上架单量",
                    "每条上架记录的上架时间作为一次容器操作完成时间",
                    "按人员和自然日排列上架完成时间；相邻操作间隔不超过阈值时，该间隔计入有效上架时长",
                    f"当前设置为 {gap_minutes} 分钟；超过阈值的空档不计入有效上架时长",
                    "上架单量 ÷ 有效上架小时（PI/小时，越高越好）",
                    "上架容器量 ÷ 有效上架小时（容器/小时，越高越好）",
                    "上架件量 ÷ 有效上架小时（件/小时，越高越好）",
                    "两个排名组分别独立排名；各组内按照上架容器量从高到低，容器量相同时依次参考上架件量和小时上架容器量",
                    "仅保留成功匹配人员主数据的员工",
                ],
            }
        )
        method.to_excel(writer, sheet_name="计算口径", index=False)

        workbook = writer.book
        fmt = _excel_formats(workbook)
        sheet = writer.sheets["上架人效排名"]
        sheet.freeze_panes(1, 0)
        sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        sheet.set_row(0, 24, fmt["header"])
        widths = [24, 10, 22, 22, 24, 14, 12, 14, 12, 12, 18, 18, 20, 18]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(1, 1, 10, fmt["integer"])
        sheet.set_column(5, 7, 14, fmt["integer"])
        sheet.set_column(8, 9, 12, fmt["decimal"])
        sheet.set_column(11, 13, 20, fmt["decimal"])

        file_sheet = writer.sheets["上传文件汇总"]
        file_sheet.set_row(0, 24, fmt["header"])
        file_sheet.set_column("A:A", 45)
        file_sheet.set_column("B:B", 14, fmt["integer"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 110)
    return output.getvalue()


def make_robot_picking_excel(result: RobotPickingResult, gap_minutes: int) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="机区拣货人效排名", index=False)
        result.file_summary.to_excel(writer, sheet_name="上传文件汇总", index=False)
        method = pd.DataFrame(
            {
                "项目": [
                    "多文件合并",
                    "考勤组",
                    "排名分组",
                    "机区拣货件量",
                    "机区任务单量",
                    "机区格口量",
                    "单件比",
                    "任务格口比",
                    "格口件比",
                    "有效机区拣货时长",
                    "连续操作阈值",
                    "人效（小时单量）",
                    "小时机区格口量",
                    "人效（小时件效）",
                    "排名规则",
                    "人员范围",
                ],
                "计算口径": [
                    "允许同时上传多个文件；仅删除因文件时间范围重叠而重复出现的明细，单个源文件内的原始行保持不变",
                    "从人员主数据中的考勤组字段匹配；源数据为空时结果保持为空",
                    "考勤组名称包含1st Shift的归入1st Shift，包含2nd Shift的归入2nd Shift，其余全部归入其他组",
                    "实际数量合计",
                    "任务单号去重数量",
                    "格口号不去重；每条格口记录计1次，同一格口号重复出现时每次均计入",
                    "机区拣货件量 ÷ 机区任务单量",
                    "机区格口量 ÷ 机区任务单量",
                    "机区拣货件量 ÷ 机区格口量",
                    "按人员和自然日排列更新时间；相邻操作间隔不超过阈值时，该间隔计入有效机区拣货时长",
                    f"当前设置为 {gap_minutes} 分钟；超过阈值的空档不计入有效机区拣货时长",
                    "机区任务单量 ÷ 有效机区拣货小时（任务单/小时，越高越好）",
                    "机区格口量 ÷ 有效机区拣货小时（格口/小时，越高越好）",
                    "机区拣货件量 ÷ 有效机区拣货小时（件/小时，越高越好）",
                    "三个排名组分别独立排名；各组内按照机区任务单量从高到低，任务单量相同时依次参考格口量、件量和小时单量",
                    "仅保留成功匹配人员主数据的员工",
                ],
            }
        )
        method.to_excel(writer, sheet_name="计算口径", index=False)

        workbook = writer.book
        fmt = _excel_formats(workbook)
        sheet = writer.sheets["机区拣货人效排名"]
        sheet.freeze_panes(1, 0)
        sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        sheet.set_row(0, 24, fmt["header"])
        widths = [16, 10, 22, 22, 22, 15, 15, 14, 12, 14, 12, 20, 18, 20, 18]
        for col_idx, width in enumerate(widths):
            sheet.set_column(col_idx, col_idx, width)
        sheet.set_column(1, 1, 10, fmt["integer"])
        sheet.set_column(5, 7, 15, fmt["integer"])
        sheet.set_column(8, 10, 14, fmt["decimal"])
        sheet.set_column(12, 14, 20, fmt["decimal"])

        file_sheet = writer.sheets["上传文件汇总"]
        file_sheet.set_row(0, 24, fmt["header"])
        file_sheet.set_column("A:A", 45)
        file_sheet.set_column("B:B", 14, fmt["integer"])

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, fmt["header"])
        method_sheet.set_column("A:A", 22)
        method_sheet.set_column("B:B", 110)
    return output.getvalue()


def render_employee_upload() -> EmployeeLookup | None:
    employee_file = st.sidebar.file_uploader(
        "1. 人员主数据",
        type=["xlsx", "xls"],
        help="验收、上架、拣货、机区拣货与打包共用。支持用户编码/Use ID、ERP、姓名和考勤组字段；考勤组为空时结果保持为空。",
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
                f"发现 {len(lookup.duplicate_keys)} 个重复匹配键，保留首次出现的姓名和考勤组。"
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

    st.subheader("验收人员排名（按验收单量）")
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
        f"排名优先按照实际完成的验收单量从高到低；"
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
- **考勤组**：从人员主数据匹配，人员表没有考勤组或该字段为空时显示为空。
- **验收件量**：验收量合计。
- **验收单量**：京东入库单号去重数量。
- **单件比**：验收件量 ÷ 验收单量。
- **系统操作时长**：原始操作区间按人员、按自然日去除重叠后相加。
- **实际验收时长**：每个PI取最早开始至最晚结束，再按人员、按自然日去除PI之间的重叠后相加。
- **人效（小时单量）**：实际验收小时 ÷ 验收单量，单位为小时/单，越低表示平均每单耗时越短。
- **排名**：优先按照实际完成的验收单量从高到低排名；单量相同再参考验收件量和平均每单用时。
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

    row2 = st.columns(5)
    row2[0].metric("总实际拣货时长", format_duration(result.total_actual_seconds))
    row2[1].metric("整体任务件比", f"{result.overall_task_piece_ratio:,.2f}")
    row2[2].metric("整体小时单量", f"{result.overall_hourly_orders:,.2f}")
    row2[3].metric("整体小时任务单量", f"{result.overall_hourly_tasks:,.2f}")
    row2[4].metric("整体小时件效", f"{result.overall_hourly_pieces:,.2f}")

    st.subheader("拣货人员分组排名（各组按订单量）")
    picking_column_config = {
        "组内排名": st.column_config.NumberColumn(format="%d"),
        "拣货件量": st.column_config.NumberColumn(format="%d"),
        "订单量": st.column_config.NumberColumn(format="%d"),
        "任务单量": st.column_config.NumberColumn(format="%d"),
        "任务件比": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时单量）": st.column_config.NumberColumn(format="%.2f"),
        "小时任务单量": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时件效）": st.column_config.NumberColumn(format="%.2f"),
    }
    for group_name in PICKING_RANK_GROUP_ORDER:
        group_ranking = result.ranking.loc[
            result.ranking["排名组"].eq(group_name)
        ].drop(columns=["排名组"])
        if group_ranking.empty:
            continue
        st.markdown(f"#### {group_name}（{len(group_ranking)}人）")
        st.dataframe(
            group_ranking,
            use_container_width=True,
            hide_index=True,
            column_config=picking_column_config,
        )

    with st.expander("查看上传文件汇总"):
        st.dataframe(result.file_summary, use_container_width=True, hide_index=True)

    duplicate_removed = result.uploaded_row_count - result.deduplicated_row_count
    st.caption(
        f"按1st Shift、2nd Shift、其他组分别排名，各组内优先按照实际完成的订单量从高到低；"
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
- **考勤组**：从人员主数据匹配，人员表没有考勤组或该字段为空时显示为空。
- **排名分组**：考勤组名称包含 `1st Shift` 的归入 **1st Shift**，包含 `2nd Shift` 的归入 **2nd Shift**，其余全部归入 **其他组**。
- **拣货件量**：实际拣货量合计。
- **订单量**：订单号去重数量。
- **任务单量**：任务单号去重数量。
- **任务件比**：拣货件量 ÷ 任务单量。
- **实际拣货时长**：每个任务从任务领取时间到该任务最晚拣货完成时间；同一人员同一天内的任务时间发生重叠时，重叠部分只计算一次。
- **人效（小时单量）**：订单量 ÷ 实际拣货小时，单位为订单/小时，越高越好。
- **小时任务单量**：任务单量 ÷ 实际拣货小时，单位为任务/小时，越高越好。
- **人效（小时件效）**：拣货件量 ÷ 实际拣货小时，单位为件/小时，越高越好。
- **排名**：三个排名组分别独立排名；每组内优先按照实际完成的订单量从高到低，订单量相同再参考拣货件量和小时单量。
- 未匹配人员不呈现，也不进入总数。
"""
        )

def render_packing_module(employee_lookup: EmployeeLookup | None) -> None:
    packing_files = st.sidebar.file_uploader(
        "2. 打包结果文件（可多选）",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="可同时上传多个时间段的文件。程序会自动合并，并去除完全重复的商品明细。",
        key="packing_data",
    )
    gap_minutes = st.sidebar.slider(
        "连续打包最大间隔（分钟）",
        min_value=1,
        max_value=60,
        value=15,
        step=1,
        help="相邻两个订单完成时间的间隔不超过该值时，计入有效打包时长。超过该值视为打包中断。",
        key="packing_gap_minutes",
    )

    if not packing_files:
        st.info("请上传人员主数据和一个或多个打包结果文件。")
        return
    if employee_lookup is None:
        st.info("请先上传人员主数据。未匹配员工不会参与结果。")
        return

    try:
        with st.spinner(f"正在合并并计算 {len(packing_files)} 个打包文件……"):
            combined, file_summary = combine_packing_files(packing_files)
            result = calculate_packing_productivity(
                combined,
                file_summary,
                employee_lookup,
                int(gap_minutes),
            )
    except Exception as exc:
        st.error(f"打包数据处理失败：{exc}")
        return

    row1 = st.columns(4)
    row1[0].metric("总打包件量", f"{result.total_pieces:,.0f}")
    row1[1].metric("总订单量", f"{result.total_orders:,}")
    row1[2].metric("整体件单比", f"{result.overall_piece_order_ratio:,.2f}")
    row1[3].metric("打包人数", f"{result.operator_count:,}")

    row2 = st.columns(4)
    row2[0].metric("总有效打包时长", format_duration(result.total_effective_seconds))
    row2[1].metric("整体小时单量", f"{result.overall_hourly_orders:,.2f}")
    row2[2].metric("整体小时件效", f"{result.overall_hourly_pieces:,.2f}")
    row2[3].metric("当前连续阈值", f"{int(gap_minutes)} 分钟")

    st.subheader("打包人员分组排名（各组按订单量）")
    packing_column_config = {
        "组内排名": st.column_config.NumberColumn(format="%d"),
        "打包件量": st.column_config.NumberColumn(format="%d"),
        "订单量": st.column_config.NumberColumn(format="%d"),
        "件单比": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时单量）": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时件效）": st.column_config.NumberColumn(format="%.2f"),
    }
    for group_name in PACKING_RANK_GROUP_ORDER:
        group_ranking = result.ranking.loc[
            result.ranking["排名组"].eq(group_name)
        ].drop(columns=["排名组"])
        if group_ranking.empty:
            continue
        st.markdown(f"#### {group_name}（{len(group_ranking)}人）")
        st.dataframe(
            group_ranking,
            use_container_width=True,
            hide_index=True,
            column_config=packing_column_config,
        )

    with st.expander("查看上传文件汇总"):
        st.dataframe(result.file_summary, use_container_width=True, hide_index=True)

    duplicate_removed = result.uploaded_row_count - result.deduplicated_row_count
    st.caption(
        f"按三个指定打包组和其他组分别排名，各组内优先按照实际完成的订单量从高到低；"
        f"上传原始记录 {result.uploaded_row_count:,} 行；"
        f"跨文件重复商品明细去除 {duplicate_removed:,} 行；"
        f"无效记录 {result.invalid_row_count:,} 行；"
        f"未匹配人员记录 {result.excluded_unmatched_rows:,} 行未呈现；"
        f"最终有效商品明细 {result.valid_row_count:,} 行。"
    )

    st.download_button(
        "下载打包人效结果 Excel",
        data=make_packing_excel(result, int(gap_minutes)),
        file_name="打包人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("打包计算口径"):
        st.markdown(
            f"""
- 可一次上传多个打包文件；程序先合并，仅删除因文件时间范围重叠而重复出现的商品明细。
- **考勤组**：从人员主数据匹配，人员表没有考勤组或该字段为空时显示为空。
- **排名分组**：`出库-打包-Babylist`、`出库-打包-Mix`、`出库-Trafilea B2B` 分别独立成组，其他考勤组全部归入 **其他组**。
- **打包件量**：实际打包件数合计。
- **订单量**：订单号去重数量，一个订单只计一次。
- **件单比**：打包件量 ÷ 订单量，用于体现平均每个订单包含的件数。
- **订单完成时间**：同一订单出现多个打包时间时，使用最晚打包时间。
- **有效打包时长**：按员工、按自然日排列订单完成时间；相邻订单间隔不超过 **{int(gap_minutes)} 分钟**时，该间隔计入有效打包时长；超过阈值的空档不计。
- **人效（小时单量）**：订单量 ÷ 有效打包小时，单位为订单/小时，越高越好。
- **人效（小时件效）**：打包件量 ÷ 有效打包小时，单位为件/小时，越高越好。
- **排名**：四个排名组分别独立排名；每组内优先按照实际完成的订单量从高到低，订单量相同再参考打包件量和小时单量。
- 原始数据没有开始打包时间，因此有效打包时长是根据相邻订单完成时间推算的工作时长。
- 未匹配人员不呈现，也不进入总数。
"""
        )

def render_putaway_module(employee_lookup: EmployeeLookup | None) -> None:
    putaway_files = st.sidebar.file_uploader(
        "2. 上架结果文件（可多选）",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="可同时上传多个时间段的文件。程序会自动合并，并去除跨文件重复明细。",
        key="putaway_data",
    )
    gap_minutes = st.sidebar.slider(
        "连续上架最大间隔（分钟）",
        min_value=1,
        max_value=60,
        value=15,
        step=1,
        help="相邻两次上架完成时间的间隔不超过该值时，计入有效上架时长。超过该值视为上架中断。",
        key="putaway_gap_minutes",
    )

    if not putaway_files:
        st.info("请上传人员主数据和一个或多个上架结果文件。")
        return
    if employee_lookup is None:
        st.info("请先上传人员主数据。未匹配员工不会参与结果。")
        return

    try:
        with st.spinner(f"正在合并 {len(putaway_files)} 个上架文件……"):
            combined, file_summary = combine_putaway_files(putaway_files)
        operation_types = sorted(
            {
                clean_name(value)
                for value in combined[PUTAWAY_COLUMNS["operation_type"]]
                if clean_name(value)
            }
        )
        operation_type = st.sidebar.selectbox(
            "上架作业类型",
            ["全部", *operation_types],
            index=0,
            key="putaway_operation_type",
            help="可单独查看采购进货、库内返架、逆向入库等作业。Robot自动上架始终排除。",
        )
        with st.spinner("正在计算上架人效……"):
            result = calculate_putaway_productivity(
                combined,
                file_summary,
                employee_lookup,
                int(gap_minutes),
                operation_type,
            )
    except Exception as exc:
        st.error(f"上架数据处理失败：{exc}")
        return

    row1 = st.columns(4)
    row1[0].metric("总上架件量", f"{result.total_pieces:,.0f}")
    row1[1].metric("总上架单量", f"{result.total_pis:,}")
    row1[2].metric("总上架容器量", f"{result.total_containers:,}")
    row1[3].metric("上架人数", f"{result.operator_count:,}")

    row2 = st.columns(6)
    row2[0].metric("总有效上架时长", format_duration(result.total_effective_seconds))
    row2[1].metric("整体容器件比", f"{result.overall_container_piece_ratio:,.2f}")
    row2[2].metric("整体PI容器比", f"{result.overall_pi_container_ratio:,.2f}")
    row2[3].metric("整体小时单量", f"{result.overall_hourly_pis:,.2f}")
    row2[4].metric("整体小时上架容器量", f"{result.overall_hourly_containers:,.2f}")
    row2[5].metric("整体小时件效", f"{result.overall_hourly_pieces:,.2f}")

    st.subheader("上架人员分组排名（各组按上架容器量）")
    putaway_column_config = {
        "组内排名": st.column_config.NumberColumn(format="%d"),
        "上架件量": st.column_config.NumberColumn(format="%d"),
        "上架单量": st.column_config.NumberColumn(format="%d"),
        "上架容器量": st.column_config.NumberColumn(format="%d"),
        "容器件比": st.column_config.NumberColumn(format="%.2f"),
        "PI容器比": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时单量）": st.column_config.NumberColumn(format="%.2f"),
        "小时上架容器量": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时件效）": st.column_config.NumberColumn(format="%.2f"),
    }
    for group_name in PUTAWAY_RANK_GROUP_ORDER:
        group_ranking = result.ranking.loc[
            result.ranking["排名组"].eq(group_name)
        ].drop(columns=["排名组"])
        if group_ranking.empty:
            continue
        st.markdown(f"#### {group_name}（{len(group_ranking)}人）")
        st.dataframe(
            group_ranking,
            use_container_width=True,
            hide_index=True,
            column_config=putaway_column_config,
        )

    with st.expander("查看上传文件汇总"):
        st.dataframe(result.file_summary, use_container_width=True, hide_index=True)

    duplicate_removed = result.uploaded_row_count - result.deduplicated_row_count
    st.caption(
        f"当前作业类型：{result.operation_type}；入库-上架组IB-Putaway与其他组分别排名，"
        f"各组内按照实际完成的上架容器量从高到低；"
        f"上传原始记录 {result.uploaded_row_count:,} 行；"
        f"跨文件重复明细去除 {duplicate_removed:,} 行；"
        f"Robot自动上架排除 {result.robot_row_count:,} 行；"
        f"无效记录 {result.invalid_row_count:,} 行；"
        f"未匹配人员记录 {result.excluded_unmatched_rows:,} 行未呈现；"
        f"最终有效明细 {result.valid_row_count:,} 行。"
    )

    st.download_button(
        "下载上架人效结果 Excel",
        data=make_putaway_excel(result, int(gap_minutes)),
        file_name="上架人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("上架计算口径"):
        st.markdown(
            f"""
- 可一次上传多个上架文件；程序先合并，仅删除因文件时间范围重叠而重复出现的明细。
- **Robot自动上架不参与计算**：储区名称等于 `Robot` 或上架员等于 `TMC_CALL` 的记录全部排除。
- **作业类型**：可选择全部，或单独查看采购进货、库内返架、逆向入库等类型。
- **考勤组**：从人员主数据匹配，人员表没有考勤组或该字段为空时显示为空。
- **排名分组**：`入库-上架组IB-Putaway` 单独成组，其余考勤组和空白全部归入 **其他组**。
- **上架件量**：上架量合计。
- **上架单量**：京东入库单号（PI）去重数量；一个PI可以包含多个容器。
- **上架容器量**：上架容器号不去重；每条有效上架记录计1次，同一容器号重复出现时每次均计入。
- **容器件比**：上架件量 ÷ 上架容器量。
- **PI容器比**：上架容器量 ÷ 上架单量。
- **有效上架时长**：按员工、按自然日排列每条上架记录的完成时间；相邻操作间隔不超过 **{int(gap_minutes)} 分钟**时，该间隔计入有效上架时长；超过阈值的空档不计。
- **人效（小时单量）**：上架单量 ÷ 有效上架小时，单位为PI/小时，越高越好。
- **小时上架容器量**：上架容器量 ÷ 有效上架小时，单位为容器/小时，越高越好。
- **人效（小时件效）**：上架件量 ÷ 有效上架小时，单位为件/小时，越高越好。
- **排名**：两个排名组分别独立排名；各组内按照上架容器量从高到低，容器量相同再参考上架件量和小时上架容器量。
- 原始数据没有开始上架时间，因此有效上架时长是根据相邻上架完成时间推算的工作时长。
- 未匹配人员不呈现，也不进入总数。
"""
        )


def render_robot_picking_module(employee_lookup: EmployeeLookup | None) -> None:
    robot_files = st.sidebar.file_uploader(
        "2. 机区拣货结果文件（可多选）",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        help="可同时上传多个时间段的机区拣货文件。程序会自动合并，并去除跨文件重复明细。",
        key="robot_picking_data",
    )
    gap_minutes = st.sidebar.slider(
        "连续机区操作最大间隔（分钟）",
        min_value=1,
        max_value=60,
        value=15,
        step=1,
        help="相邻两次更新时间的间隔不超过该值时，计入有效机区拣货时长。超过该值视为作业中断。",
        key="robot_picking_gap_minutes",
    )

    if not robot_files:
        st.info("请上传人员主数据和一个或多个机区拣货结果文件。")
        return
    if employee_lookup is None:
        st.info("请先上传人员主数据。未匹配员工不会参与结果。")
        return

    try:
        with st.spinner(f"正在合并并计算 {len(robot_files)} 个机区拣货文件……"):
            combined, file_summary = combine_robot_picking_files(robot_files)
            result = calculate_robot_picking_productivity(
                combined,
                file_summary,
                employee_lookup,
                int(gap_minutes),
            )
    except Exception as exc:
        st.error(f"机区拣货数据处理失败：{exc}")
        return

    row1 = st.columns(4)
    row1[0].metric("总机区拣货件量", f"{result.total_pieces:,.0f}")
    row1[1].metric("总机区任务单量", f"{result.total_tasks:,}")
    row1[2].metric("总机区格口量", f"{result.total_slots:,}")
    row1[3].metric("机区拣货人数", f"{result.operator_count:,}")

    row2 = st.columns(6)
    row2[0].metric("总有效机区拣货时长", format_duration(result.total_effective_seconds))
    row2[1].metric("整体单件比", f"{result.overall_piece_task_ratio:,.2f}")
    row2[2].metric("整体任务格口比", f"{result.overall_task_slot_ratio:,.2f}")
    row2[3].metric("整体格口件比", f"{result.overall_slot_piece_ratio:,.2f}")
    row2[4].metric("整体小时任务单量", f"{result.overall_hourly_tasks:,.2f}")
    row2[5].metric("整体小时机区格口量", f"{result.overall_hourly_slots:,.2f}")
    st.metric("整体小时件效", f"{result.overall_hourly_pieces:,.2f}")

    st.subheader("机区拣货人员分组排名（各组按任务单量）")
    robot_column_config = {
        "组内排名": st.column_config.NumberColumn(format="%d"),
        "机区拣货件量": st.column_config.NumberColumn(format="%d"),
        "机区任务单量": st.column_config.NumberColumn(format="%d"),
        "机区格口量": st.column_config.NumberColumn(format="%d"),
        "单件比": st.column_config.NumberColumn(format="%.2f"),
        "任务格口比": st.column_config.NumberColumn(format="%.2f"),
        "格口件比": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时单量）": st.column_config.NumberColumn(format="%.2f"),
        "小时机区格口量": st.column_config.NumberColumn(format="%.2f"),
        "人效（小时件效）": st.column_config.NumberColumn(format="%.2f"),
    }
    for group_name in PICKING_RANK_GROUP_ORDER:
        group_ranking = result.ranking.loc[
            result.ranking["排名组"].eq(group_name)
        ].drop(columns=["排名组"])
        if group_ranking.empty:
            continue
        st.markdown(f"#### {group_name}（{len(group_ranking)}人）")
        st.dataframe(
            group_ranking,
            use_container_width=True,
            hide_index=True,
            column_config=robot_column_config,
        )

    with st.expander("查看上传文件汇总"):
        st.dataframe(result.file_summary, use_container_width=True, hide_index=True)

    duplicate_removed = result.uploaded_row_count - result.deduplicated_row_count
    st.caption(
        f"按1st Shift、2nd Shift、其他组分别排名，各组内优先按照实际完成的机区任务单量从高到低；"
        f"格口号不去重，每条有效格口记录均计1次；当前连续阈值为 {int(gap_minutes)} 分钟；"
        f"上传原始记录 {result.uploaded_row_count:,} 行；"
        f"跨文件重复明细去除 {duplicate_removed:,} 行；"
        f"无效记录 {result.invalid_row_count:,} 行；"
        f"未匹配人员记录 {result.excluded_unmatched_rows:,} 行未呈现；"
        f"最终有效记录 {result.valid_row_count:,} 行。"
    )

    st.download_button(
        "下载机区拣货人效结果 Excel",
        data=make_robot_picking_excel(result, int(gap_minutes)),
        file_name="机区拣货人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("机区拣货计算口径"):
        st.markdown(
            f"""
- 可一次上传多个机区拣货文件；程序先合并，仅删除因文件时间范围重叠而重复出现的明细。
- **人员匹配**：使用更新人账号去除 `@jd.com` 后匹配人员主数据，同时带出实际姓名和考勤组。
- **排名分组**：考勤组名称包含 `1st Shift` 的归入 **1st Shift**，包含 `2nd Shift` 的归入 **2nd Shift**，其余全部归入 **其他组**。
- **机区拣货件量**：实际数量合计。
- **机区任务单量**：任务单号去重数量。
- **机区格口量**：使用格口号，不去重；每条有效格口记录计1次，同一格口号重复出现时每次均计入。
- **单件比**：机区拣货件量 ÷ 机区任务单量。
- **任务格口比**：机区格口量 ÷ 机区任务单量。
- **格口件比**：机区拣货件量 ÷ 机区格口量。
- **有效机区拣货时长**：按员工、按自然日排列更新时间；相邻操作间隔不超过 **{int(gap_minutes)} 分钟**时，该间隔计入有效时长；超过阈值的空档不计。
- **人效（小时单量）**：机区任务单量 ÷ 有效机区拣货小时，单位为任务单/小时，越高越好。
- **小时机区格口量**：机区格口量 ÷ 有效机区拣货小时，单位为格口/小时，越高越好。
- **人效（小时件效）**：机区拣货件量 ÷ 有效机区拣货小时，单位为件/小时，越高越好。
- **排名**：三个排名组分别独立排名；各组内按照机区任务单量从高到低，任务单量相同再依次参考格口量、件量和小时单量。
- 原始数据没有开始时间，因此有效机区拣货时长是根据相邻更新时间推算的工作时长。
- 未匹配人员不呈现，也不进入总数。
"""
        )


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.caption("统一人员主数据｜验收、上架、拣货、机区拣货与打包人效分析｜结果可下载为Excel")

    with st.sidebar:
        st.header("业务模块")
        module = st.radio(
            "选择分析环节",
            ["验收", "上架", "拣货", "机区拣货", "打包"],
            horizontal=True,
        )
        st.divider()
        st.header("数据上传")

    employee_lookup = render_employee_upload()

    if module == "验收":
        render_acceptance_module(employee_lookup)
    elif module == "上架":
        render_putaway_module(employee_lookup)
    elif module == "拣货":
        render_picking_module(employee_lookup)
    elif module == "机区拣货":
        render_robot_picking_module(employee_lookup)
    else:
        render_packing_module(employee_lookup)


if __name__ == "__main__":
    render_app()
