from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "LAX2 人效分析工具"

ACCEPTANCE_COLUMNS = {
    "order": "京东入库单号",
    "quantity": "验收量",
    "operator": "验收人",
    "start": "开始验收时间",
    "end": "最后验收时间",
}

EMPLOYEE_COLUMN_ALIASES = {
    "user_id": ["用户编码", "Use ID", "UseID", "User ID", "userid", "use id"],
    "erp": ["ERP", "erp"],
    "name": ["姓名", "员工姓名", "Name", "Employee Name"],
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
    total_quantity: float
    total_order_count: int
    total_effective_seconds: float
    overall_hourly_productivity: float
    overall_order_productivity: float
    operator_count: int
    included_row_count: int
    total_row_count: int


def normalize_column_label(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def find_column(columns: Iterable[object], aliases: list[str]) -> str | None:
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


def extract_operator_id(value: object) -> str:
    """将验收人字段转换为人员主数据可匹配的账号。"""
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


def match_employee(operator_id: str, lookup: EmployeeLookup) -> str:
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


def merge_overlapping_seconds(group: pd.DataFrame) -> float:
    """按人员、按自然日去重重叠时间；不同日期分别计算。"""
    daily_intervals: dict[object, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}

    for start, end in group[["开始时间", "结束时间"]].itertuples(
        index=False, name=None
    ):
        # 跨午夜的记录拆分到各自自然日，确保只在同一天内去重。
        segment_start = start
        while segment_start.normalize() < end.normalize():
            next_midnight = segment_start.normalize() + pd.Timedelta(days=1)
            daily_intervals.setdefault(segment_start.date(), []).append(
                (segment_start, next_midnight)
            )
            segment_start = next_midnight

        if end > segment_start:
            daily_intervals.setdefault(segment_start.date(), []).append(
                (segment_start, end)
            )

    total_seconds = 0.0

    for intervals in daily_intervals.values():
        intervals.sort(key=lambda item: (item[0], item[1]))
        current_start: pd.Timestamp | None = None
        current_end: pd.Timestamp | None = None

        for start, end in intervals:
            if current_start is None:
                current_start, current_end = start, end
                continue

            # 仅合并同一自然日内重叠或首尾相接的区间。
            if start <= current_end:
                if end > current_end:
                    current_end = end
            else:
                total_seconds += (current_end - current_start).total_seconds()
                current_start, current_end = start, end

        if current_start is not None and current_end is not None:
            total_seconds += (current_end - current_start).total_seconds()

    return float(total_seconds)


def count_unique_orders(series: pd.Series) -> int:
    cleaned = series.fillna("").astype(str).str.strip()
    return int(cleaned[cleaned.ne("")].nunique())


def calculate_acceptance_productivity(
    acceptance_df: pd.DataFrame,
    employee_lookup: EmployeeLookup,
) -> AcceptanceResult:
    missing_columns = [
        column for column in ACCEPTANCE_COLUMNS.values() if column not in acceptance_df.columns
    ]
    if missing_columns:
        raise ValueError("验收表缺少必要字段：" + "、".join(missing_columns))

    work = acceptance_df[list(ACCEPTANCE_COLUMNS.values())].copy()
    work["验收人账号"] = work[ACCEPTANCE_COLUMNS["operator"]].map(extract_operator_id)
    work["实际姓名"] = work["验收人账号"].map(
        lambda value: match_employee(value, employee_lookup)
    )
    work["京东入库单号清洗"] = (
        work[ACCEPTANCE_COLUMNS["order"]].fillna("").astype(str).str.strip()
    )
    work["验收量数值"] = pd.to_numeric(
        work[ACCEPTANCE_COLUMNS["quantity"]], errors="coerce"
    )
    work["开始时间"] = pd.to_datetime(
        work[ACCEPTANCE_COLUMNS["start"]], errors="coerce"
    )
    work["结束时间"] = pd.to_datetime(
        work[ACCEPTANCE_COLUMNS["end"]], errors="coerce"
    )

    valid_mask = (
        work["验收人账号"].ne("")
        & work["实际姓名"].ne("")
        & work["验收量数值"].notna()
        & work["开始时间"].notna()
        & work["结束时间"].notna()
        & (work["结束时间"] >= work["开始时间"])
    )
    valid = work.loc[valid_mask].copy()

    if valid.empty:
        raise ValueError("没有可用于计算的人效记录，请检查人员表、验收量和时间字段。")

    ranking_rows: list[dict[str, object]] = []
    for (operator_id, employee_name), group in valid.groupby(
        ["验收人账号", "实际姓名"], sort=False
    ):
        ranking_rows.append(
            {
                "验收人账号": operator_id,
                "实际姓名": employee_name,
                "验收量": float(group["验收量数值"].sum()),
                "验收单量": count_unique_orders(group["京东入库单号清洗"]),
                "有效秒数": merge_overlapping_seconds(group),
            }
        )

    ranking = pd.DataFrame(ranking_rows)
    ranking["有效工作小时"] = ranking["有效秒数"] / 3600
    ranking["小时人效"] = np.where(
        ranking["有效工作小时"] > 0,
        ranking["验收量"] / ranking["有效工作小时"],
        np.nan,
    )
    ranking["人效（单量）"] = np.where(
        ranking["有效工作小时"] > 0,
        ranking["验收单量"] / ranking["有效工作小时"],
        np.nan,
    )

    ranking = ranking.sort_values(
        ["小时人效", "验收量"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    ranking.insert(0, "排名", np.arange(1, len(ranking) + 1))
    ranking["总有效工作时长"] = ranking["有效秒数"].map(format_duration)

    total_quantity = float(ranking["验收量"].sum())
    total_order_count = count_unique_orders(valid["京东入库单号清洗"])
    total_effective_seconds = float(ranking["有效秒数"].sum())
    total_effective_hours = total_effective_seconds / 3600

    overall_hourly_productivity = (
        total_quantity / total_effective_hours
        if total_effective_hours > 0
        else float("nan")
    )
    overall_order_productivity = (
        total_order_count / total_effective_hours
        if total_effective_hours > 0
        else float("nan")
    )

    ranking_display = ranking[
        [
            "排名",
            "验收人账号",
            "实际姓名",
            "验收量",
            "验收单量",
            "总有效工作时长",
            "小时人效",
            "人效（单量）",
        ]
    ].copy()

    return AcceptanceResult(
        ranking=ranking_display,
        total_quantity=total_quantity,
        total_order_count=total_order_count,
        total_effective_seconds=total_effective_seconds,
        overall_hourly_productivity=overall_hourly_productivity,
        overall_order_productivity=overall_order_productivity,
        operator_count=len(ranking_display),
        included_row_count=len(valid),
        total_row_count=len(work),
    )


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes) -> list[str]:
    with pd.ExcelFile(io.BytesIO(file_bytes)) as workbook:
        return workbook.sheet_names


@st.cache_data(show_spinner=False)
def read_excel_sheet(
    file_bytes: bytes,
    sheet_name: str,
    usecols: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    selected_columns = list(usecols) if usecols else None
    return pd.read_excel(
        io.BytesIO(file_bytes),
        sheet_name=sheet_name,
        usecols=selected_columns,
    )


def make_excel_output(result: AcceptanceResult) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="验收人效排名", index=False)

        method = pd.DataFrame(
            {
                "项目": [
                    "验收量",
                    "验收单量",
                    "个人总有效工作时长",
                    "小时人效",
                    "人效（单量）",
                    "人员匹配",
                    "总验收单量",
                    "整体总有效工作时长",
                ],
                "计算口径": [
                    "按验收人汇总计入计算记录中的验收量",
                    "按验收人对京东入库单号去重计数；同一人员重复出现的同一单号只计1单",
                    "按验收人、按自然日合并开始至结束时间区间；只去重同一天内的重叠时间，不使用额外间隔阈值",
                    "验收量 ÷ 有效工作小时",
                    "验收单量 ÷ 有效工作小时",
                    "先匹配用户编码/Use ID，再匹配ERP；账号中的@域名自动移除；未匹配人员不呈现且不计入结果",
                    "所有已匹配记录中的京东入库单号全局去重计数",
                    "所有已匹配验收人的个人总有效工作时长相加",
                ],
            }
        )
        method.to_excel(writer, sheet_name="计算口径", index=False)

        workbook = writer.book
        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "border": 1,
                "align": "center",
                "valign": "vcenter",
            }
        )
        integer_format = workbook.add_format({"num_format": "#,##0"})
        decimal_format = workbook.add_format({"num_format": "#,##0.00"})
        text_format = workbook.add_format({"valign": "vcenter"})

        ranking_sheet = writer.sheets["验收人效排名"]
        ranking_sheet.freeze_panes(1, 0)
        ranking_sheet.autofilter(
            0, 0, len(result.ranking), len(result.ranking.columns) - 1
        )
        ranking_sheet.set_row(0, 24, header_format)
        ranking_sheet.set_column("A:A", 8, integer_format)
        ranking_sheet.set_column("B:B", 24, text_format)
        ranking_sheet.set_column("C:C", 22, text_format)
        ranking_sheet.set_column("D:E", 14, integer_format)
        ranking_sheet.set_column("F:F", 20, text_format)
        ranking_sheet.set_column("G:H", 16, decimal_format)

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, header_format)
        method_sheet.set_column("A:A", 24)
        method_sheet.set_column("B:B", 100)
        method_sheet.set_default_row(30)

    return output.getvalue()


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.caption("第一阶段：验收人效｜人员主数据可复用于后续所有业务模块")

    with st.sidebar:
        st.header("数据上传")
        employee_file = st.file_uploader(
            "1. 人员主数据",
            type=["xlsx", "xls"],
            help="用于将系统账号匹配为实际姓名。支持用户编码/Use ID、ERP和姓名字段。",
            key="employee_master",
        )
        acceptance_file = st.file_uploader(
            "2. 验收明细表",
            type=["xlsx", "xls"],
            key="acceptance_data",
        )

        st.divider()
        st.markdown("**后续模块**")
        st.caption("上架、复核、打包、大波次拣货、普通拣货将继续使用同一张人员主数据匹配姓名。")

    if employee_file is None or acceptance_file is None:
        st.info("请上传人员主数据和验收明细表。")
        return

    try:
        employee_bytes = employee_file.getvalue()
        employee_sheets = get_sheet_names(employee_bytes)
        employee_sheet = st.sidebar.selectbox(
            "人员表工作表",
            employee_sheets,
            key="employee_sheet",
        )
        employee_df = read_excel_sheet(employee_bytes, employee_sheet)
        employee_lookup = build_employee_lookup(employee_df)
        st.sidebar.success(f"人员源数据已读取：{employee_lookup.source_rows:,} 行")
        if employee_lookup.duplicate_keys:
            st.sidebar.warning(
                f"发现 {len(employee_lookup.duplicate_keys)} 个重复匹配键，程序保留首次出现的姓名。"
            )

        acceptance_bytes = acceptance_file.getvalue()
        acceptance_sheets = get_sheet_names(acceptance_bytes)
        acceptance_sheet = st.sidebar.selectbox(
            "验收表工作表",
            acceptance_sheets,
            key="acceptance_sheet",
        )

        with st.spinner("正在读取并计算验收人效……"):
            acceptance_df = read_excel_sheet(
                acceptance_bytes,
                acceptance_sheet,
                tuple(ACCEPTANCE_COLUMNS.values()),
            )
            result = calculate_acceptance_productivity(
                acceptance_df=acceptance_df,
                employee_lookup=employee_lookup,
            )
    except Exception as exc:
        st.error(f"处理失败：{exc}")
        return

    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("总验收量", f"{result.total_quantity:,.0f}")
    kpi2.metric("总验收单量", f"{result.total_order_count:,}")
    kpi3.metric("总有效工作时长", format_duration(result.total_effective_seconds))

    kpi4, kpi5, kpi6 = st.columns(3)
    kpi4.metric("整体小时人效", f"{result.overall_hourly_productivity:,.2f}")
    kpi5.metric("整体单量人效", f"{result.overall_order_productivity:,.2f}")
    kpi6.metric("验收人数", f"{result.operator_count:,}")

    st.subheader("验收人员小时人效排名")
    st.dataframe(
        result.ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "排名": st.column_config.NumberColumn(format="%d"),
            "验收量": st.column_config.NumberColumn(format="%d"),
            "验收单量": st.column_config.NumberColumn(format="%d"),
            "小时人效": st.column_config.NumberColumn(format="%.2f"),
            "人效（单量）": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    st.caption(f"计入计算的有效记录：{result.included_row_count:,} 行。")

    output_bytes = make_excel_output(result)
    st.download_button(
        "下载验收人效结果 Excel",
        data=output_bytes,
        file_name="验收人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("计算口径"):
        st.markdown(
            """
- **验收量**：按验收人汇总计入计算记录中的验收量。
- **验收单量**：按人员统计唯一的“京东入库单号”；同一人员重复出现的同一单号只计1单。
- **验收人账号**：自动去除括号和邮箱域名，例如 `US018958(US018958@jd.com)` 转为 `US018958`。
- **实际姓名**：先匹配人员表中的用户编码/Use ID，再匹配 ERP；未匹配人员不呈现且不计入结果。
- **个人总有效工作时长**：按验收人、按自然日处理时间区间；只去重发生在同一天内的重叠时间，不同日期分别计算；不使用额外间隔阈值。
- **小时人效**：验收量 ÷ 有效工作小时。
- **人效（单量）**：验收单量 ÷ 有效工作小时。
- **总验收单量**：所有已匹配记录中的京东入库单号全局去重计数。
- **总有效工作时长**：所有已匹配验收人的个人总有效工作时长相加。
"""
        )


if __name__ == "__main__":
    render_app()
