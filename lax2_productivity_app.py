from __future__ import annotations

import io
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st


APP_TITLE = "LAX2 人效分析工具"

ACCEPTANCE_COLUMNS = {
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
    unmatched: pd.DataFrame
    invalid_rows: pd.DataFrame
    total_quantity: float
    total_effective_seconds: float
    overall_hourly_productivity: float
    operator_count: int
    valid_row_count: int
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
    """Convert operator text to an account ID suitable for employee lookup.

    Examples:
    US018958(US018958@jd.com) -> US018958
    fang.yang@jd.com          -> fang.yang
    huiyingsun369@gmail.com   -> huiyingsun369
    """
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
            "人员源数据中未找到姓名列。支持的列名包括：姓名、员工姓名、Name、Employee Name。"
        )
    if user_id_col is None and erp_col is None:
        raise ValueError(
            "人员源数据中未找到用户编码/Use ID或ERP列，无法匹配人员。"
        )

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


def match_employee(operator_id: str, lookup: EmployeeLookup | None) -> tuple[str, str]:
    if lookup is None:
        return "未上传人员表", "未匹配"

    exact = canonical_key(operator_id)
    compact = compact_key(operator_id)

    if exact in lookup.exact_user_id:
        return lookup.exact_user_id[exact], "用户编码"
    if exact in lookup.exact_erp:
        return lookup.exact_erp[exact], "ERP"
    if compact and compact in lookup.normalized_user_id:
        return lookup.normalized_user_id[compact], "用户编码（标准化）"
    if compact and compact in lookup.normalized_erp:
        return lookup.normalized_erp[compact], "ERP（标准化）"
    return "未匹配", "未匹配"


def merge_intervals_effective_seconds(
    starts: pd.Series,
    ends: pd.Series,
    gap_minutes: int,
) -> float:
    intervals = sorted(zip(starts.tolist(), ends.tolist()), key=lambda item: item[0])
    if not intervals:
        return 0.0

    allowed_gap = pd.Timedelta(minutes=gap_minutes)
    current_start, current_end = intervals[0]
    total_seconds = 0.0

    for next_start, next_end in intervals[1:]:
        if next_start <= current_end + allowed_gap:
            if next_end > current_end:
                current_end = next_end
        else:
            total_seconds += (current_end - current_start).total_seconds()
            current_start, current_end = next_start, next_end

    total_seconds += (current_end - current_start).total_seconds()
    return max(total_seconds, 0.0)


def format_duration(total_seconds: float) -> str:
    rounded = max(int(round(total_seconds)), 0)
    hours, remainder = divmod(rounded, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def calculate_acceptance_productivity(
    acceptance_df: pd.DataFrame,
    employee_lookup: EmployeeLookup | None,
    gap_minutes: int = 5,
) -> AcceptanceResult:
    missing_columns = [
        column for column in ACCEPTANCE_COLUMNS.values() if column not in acceptance_df.columns
    ]
    if missing_columns:
        raise ValueError("验收表缺少必要字段：" + "、".join(missing_columns))

    work = acceptance_df[list(ACCEPTANCE_COLUMNS.values())].copy()
    work["原始行号"] = np.arange(2, len(work) + 2)
    work["验收人账号"] = work[ACCEPTANCE_COLUMNS["operator"]].map(extract_operator_id)
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
        & work["验收量数值"].notna()
        & work["开始时间"].notna()
        & work["结束时间"].notna()
        & (work["结束时间"] >= work["开始时间"])
    )

    invalid_rows = work.loc[~valid_mask].copy()
    valid = work.loc[valid_mask].copy()

    if valid.empty:
        raise ValueError("没有可用于计算的人效记录，请检查验收人、验收量和时间字段。")

    valid["工作日期"] = valid["开始时间"].dt.date

    quantity_summary = (
        valid.groupby("验收人账号", as_index=False)
        .agg(验收量=("验收量数值", "sum"), 有效记录数=("验收量数值", "size"))
    )

    duration_records: list[dict[str, object]] = []
    for (operator_id, work_date), group in valid.groupby(
        ["验收人账号", "工作日期"], sort=False
    ):
        duration_records.append(
            {
                "验收人账号": operator_id,
                "工作日期": work_date,
                "有效秒数": merge_intervals_effective_seconds(
                    group["开始时间"], group["结束时间"], gap_minutes
                ),
            }
        )

    daily_duration = pd.DataFrame(duration_records)
    duration_summary = (
        daily_duration.groupby("验收人账号", as_index=False)["有效秒数"].sum()
    )

    ranking = quantity_summary.merge(duration_summary, on="验收人账号", how="left")
    ranking["有效工作小时"] = ranking["有效秒数"] / 3600
    ranking["小时人效"] = np.where(
        ranking["有效工作小时"] > 0,
        ranking["验收量"] / ranking["有效工作小时"],
        np.nan,
    )

    matched_values = ranking["验收人账号"].apply(
        lambda value: match_employee(value, employee_lookup)
    )
    ranking["实际姓名"] = matched_values.map(lambda value: value[0])
    ranking["匹配方式"] = matched_values.map(lambda value: value[1])
    ranking["匹配状态"] = np.where(
        ranking["匹配方式"].eq("未匹配"), "未匹配", "已匹配"
    )

    ranking = ranking.sort_values(
        ["小时人效", "验收量"], ascending=[False, False], na_position="last"
    ).reset_index(drop=True)
    ranking.insert(0, "排名", np.arange(1, len(ranking) + 1))
    ranking["总有效工作时长"] = ranking["有效秒数"].map(format_duration)

    total_quantity = float(ranking["验收量"].sum())
    total_effective_seconds = float(ranking["有效秒数"].sum())
    overall_hourly_productivity = (
        total_quantity / (total_effective_seconds / 3600)
        if total_effective_seconds > 0
        else float("nan")
    )

    display_columns = [
        "排名",
        "验收人账号",
        "实际姓名",
        "验收量",
        "总有效工作时长",
        "小时人效",
        "匹配状态",
    ]
    ranking_display = ranking[display_columns].copy()

    unmatched = ranking.loc[
        ranking["匹配状态"].eq("未匹配"),
        ["验收人账号", "验收量", "总有效工作时长", "小时人效"],
    ].copy()

    return AcceptanceResult(
        ranking=ranking_display,
        unmatched=unmatched,
        invalid_rows=invalid_rows,
        total_quantity=total_quantity,
        total_effective_seconds=total_effective_seconds,
        overall_hourly_productivity=overall_hourly_productivity,
        operator_count=len(ranking),
        valid_row_count=len(valid),
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


def make_excel_output(result: AcceptanceResult, gap_minutes: int) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="验收人效排名", index=False)

        if result.unmatched.empty:
            pd.DataFrame({"提示": ["所有验收人均已匹配实际姓名"]}).to_excel(
                writer, sheet_name="未匹配人员", index=False
            )
        else:
            result.unmatched.to_excel(writer, sheet_name="未匹配人员", index=False)

        method = pd.DataFrame(
            {
                "项目": [
                    "验收量",
                    "有效工作时长",
                    "连续工作间隔",
                    "小时人效",
                    "人员匹配",
                    "总有效工作时长",
                ],
                "计算口径": [
                    "按验收人汇总有效记录中的验收量",
                    "按验收人和日期合并重叠时间区间；相邻记录间隔在阈值内视为连续工作",
                    f"当前设置为 {gap_minutes} 分钟；超过该时间的空档不计入有效工作时长",
                    "验收量 ÷ 有效工作小时",
                    "先匹配用户编码/Use ID，再匹配ERP；账号中的@域名会自动移除",
                    "所有验收人的有效工作时长相加",
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
        unmatched_format = workbook.add_format(
            {"font_color": "#9C0006", "bg_color": "#FFC7CE"}
        )

        ranking_sheet = writer.sheets["验收人效排名"]
        ranking_sheet.freeze_panes(1, 0)
        ranking_sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        ranking_sheet.set_row(0, 24, header_format)
        ranking_sheet.set_column("A:A", 8, integer_format)
        ranking_sheet.set_column("B:B", 24, text_format)
        ranking_sheet.set_column("C:C", 22, text_format)
        ranking_sheet.set_column("D:D", 14, integer_format)
        ranking_sheet.set_column("E:E", 20, text_format)
        ranking_sheet.set_column("F:F", 14, decimal_format)
        ranking_sheet.set_column("G:G", 12, text_format)
        if len(result.ranking) > 0:
            ranking_sheet.conditional_format(
                1,
                6,
                len(result.ranking),
                6,
                {
                    "type": "text",
                    "criteria": "containing",
                    "value": "未匹配",
                    "format": unmatched_format,
                },
            )

        unmatched_sheet = writer.sheets["未匹配人员"]
        unmatched_sheet.freeze_panes(1, 0)
        unmatched_sheet.set_row(0, 24, header_format)
        unmatched_sheet.set_column("A:A", 26)
        unmatched_sheet.set_column("B:B", 14, integer_format)
        unmatched_sheet.set_column("C:C", 20)
        unmatched_sheet.set_column("D:D", 14, decimal_format)

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, header_format)
        method_sheet.set_column("A:A", 20)
        method_sheet.set_column("B:B", 90)
        method_sheet.set_default_row(28)

    return output.getvalue()


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.caption("第一阶段：验收小时人效｜人员主数据可复用于后续所有业务模块")

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
        gap_minutes = st.number_input(
            "连续工作最大间隔（分钟）",
            min_value=0,
            max_value=60,
            value=5,
            step=1,
            help="相邻记录间隔不超过该值，视为同一段连续工作；超过的空档不计入有效工作时长。",
        )

        st.divider()
        st.markdown("**后续模块**")
        st.caption("上架、复核、打包、大波次拣货、普通拣货将继续使用同一张人员主数据匹配姓名。")

    employee_lookup: EmployeeLookup | None = None

    if employee_file is not None:
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
        except Exception as exc:
            st.sidebar.error(f"人员表读取失败：{exc}")
            employee_lookup = None
    else:
        st.info("请先上传人员主数据和验收明细表。人员表更新后可直接重新上传，无需修改程序。")

    if acceptance_file is None:
        return

    try:
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
                gap_minutes=int(gap_minutes),
            )
    except Exception as exc:
        st.error(f"处理失败：{exc}")
        return

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("总验收量", f"{result.total_quantity:,.0f}")
    kpi2.metric("总有效工作时长", format_duration(result.total_effective_seconds))
    kpi3.metric("整体小时人效", f"{result.overall_hourly_productivity:,.2f}")
    kpi4.metric("验收人数", f"{result.operator_count:,}")

    st.subheader("验收人员小时人效排名")
    st.dataframe(
        result.ranking,
        use_container_width=True,
        hide_index=True,
        column_config={
            "排名": st.column_config.NumberColumn(format="%d"),
            "验收量": st.column_config.NumberColumn(format="%d"),
            "小时人效": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    matched_count = result.operator_count - len(result.unmatched)
    if result.unmatched.empty:
        st.success(f"人员姓名匹配完成：{matched_count}/{result.operator_count}。")
    else:
        st.warning(
            f"人员姓名已匹配 {matched_count}/{result.operator_count}；"
            f"另有 {len(result.unmatched)} 个账号未在人员源数据中找到。"
        )
        with st.expander("查看未匹配人员", expanded=True):
            st.dataframe(result.unmatched, use_container_width=True, hide_index=True)

    if not result.invalid_rows.empty:
        st.warning(
            f"共有 {len(result.invalid_rows):,} 行因账号、验收量或时间无效而未计入人效。"
        )

    st.caption(
        f"有效记录：{result.valid_row_count:,}/{result.total_row_count:,} 行。"
        f"当前连续工作间隔阈值：{int(gap_minutes)} 分钟。"
    )

    output_bytes = make_excel_output(result, int(gap_minutes))
    st.download_button(
        "下载验收人效结果 Excel",
        data=output_bytes,
        file_name="验收人效排名结果.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

    with st.expander("计算口径"):
        st.markdown(
            f"""
- **验收量**：按验收人汇总有效记录中的验收量。
- **验收人账号**：自动从账号中去除括号和邮箱域名，例如 `US018958(US018958@jd.com)` 转为 `US018958`。
- **实际姓名**：先匹配人员表中的用户编码/Use ID，再匹配 ERP。
- **有效工作时长**：按验收人和日期合并重叠时间；相邻记录间隔不超过 **{int(gap_minutes)} 分钟**时视为连续工作。
- **小时人效**：验收量 ÷ 有效工作小时。
- **总有效工作时长**：所有验收人的有效工作时长相加。
"""
        )


if __name__ == "__main__":
    render_app()
