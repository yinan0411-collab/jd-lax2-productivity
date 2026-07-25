from __future__ import annotations

import hmac
import io
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from supabase import Client, create_client


APP_TITLE = "LAX2 人效分析工具"
EMPLOYEE_TABLE = "employee_master"

ACCEPTANCE_COLUMNS = {
    "quantity": "验收量",
    "operator": "验收人",
    "start": "开始验收时间",
    "end": "最后验收时间",
}

EMPLOYEE_COLUMN_ALIASES = {
    "user_id": [
        "用户编码",
        "Use ID",
        "UseID",
        "User ID",
        "userid",
        "use id",
        "user_id",
    ],
    "erp": ["ERP", "erp"],
    "name": ["姓名", "员工姓名", "Name", "Employee Name", "name"],
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
    """将账号转换成可用于人员匹配的 ID。"""
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


def build_record_key(user_id: str, erp: str) -> str:
    if compact_key(user_id):
        return f"uid:{compact_key(user_id)}"
    if compact_key(erp):
        return f"erp:{compact_key(erp)}"
    return ""


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
        raise ValueError("人员主数据中未找到姓名列。")
    if user_id_col is None and erp_col is None:
        raise ValueError("人员主数据中未找到用户编码/Use ID或ERP列。")

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
            raw_user_id = extract_operator_id(row.get(user_id_col))
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
            raw_erp = extract_operator_id(row.get(erp_col))
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


def prepare_employee_import(employee_df: pd.DataFrame) -> pd.DataFrame:
    user_id_col = find_column(employee_df.columns, EMPLOYEE_COLUMN_ALIASES["user_id"])
    erp_col = find_column(employee_df.columns, EMPLOYEE_COLUMN_ALIASES["erp"])
    name_col = find_column(employee_df.columns, EMPLOYEE_COLUMN_ALIASES["name"])

    if name_col is None:
        raise ValueError(
            "上传表中未找到姓名列。支持：姓名、员工姓名、Name、Employee Name。"
        )
    if user_id_col is None and erp_col is None:
        raise ValueError("上传表中至少需要用户编码/Use ID或ERP列。")

    records: list[dict[str, str]] = []
    for _, row in employee_df.iterrows():
        user_id = extract_operator_id(row.get(user_id_col)) if user_id_col else ""
        erp = extract_operator_id(row.get(erp_col)) if erp_col else ""
        name = clean_name(row.get(name_col))
        record_key = build_record_key(user_id, erp)
        if not name or not record_key:
            continue
        records.append(
            {
                "record_key": record_key,
                "user_id": user_id,
                "erp": erp,
                "name": name,
            }
        )

    if not records:
        raise ValueError("没有找到可导入的有效人员记录。")

    result = pd.DataFrame(records)
    return result.drop_duplicates(subset=["record_key"], keep="last").reset_index(
        drop=True
    )


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
    employee_lookup: EmployeeLookup,
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
    work["开始时间"] = pd.to_datetime(work[ACCEPTANCE_COLUMNS["start"]], errors="coerce")
    work["结束时间"] = pd.to_datetime(work[ACCEPTANCE_COLUMNS["end"]], errors="coerce")

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

    valid["实际姓名"] = valid["验收人账号"].map(
        lambda value: match_employee(value, employee_lookup)
    )
    # 未匹配人员不在页面、排名、KPI和下载结果中呈现。
    valid = valid.loc[valid["实际姓名"].ne("")].copy()

    if valid.empty:
        raise ValueError("验收表中的账号均未在人员主数据中匹配到实际姓名。")

    valid["工作日期"] = valid["开始时间"].dt.date

    quantity_summary = (
        valid.groupby(["验收人账号", "实际姓名"], as_index=False)
        .agg(验收量=("验收量数值", "sum"), 有效记录数=("验收量数值", "size"))
    )

    duration_records: list[dict[str, object]] = []
    for (operator_id, actual_name, work_date), group in valid.groupby(
        ["验收人账号", "实际姓名", "工作日期"], sort=False
    ):
        duration_records.append(
            {
                "验收人账号": operator_id,
                "实际姓名": actual_name,
                "工作日期": work_date,
                "有效秒数": merge_intervals_effective_seconds(
                    group["开始时间"], group["结束时间"], gap_minutes
                ),
            }
        )

    daily_duration = pd.DataFrame(duration_records)
    duration_summary = daily_duration.groupby(
        ["验收人账号", "实际姓名"], as_index=False
    )["有效秒数"].sum()

    ranking = quantity_summary.merge(
        duration_summary, on=["验收人账号", "实际姓名"], how="left"
    )
    ranking["有效工作小时"] = ranking["有效秒数"] / 3600
    ranking["小时人效"] = np.where(
        ranking["有效工作小时"] > 0,
        ranking["验收量"] / ranking["有效工作小时"],
        np.nan,
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
    ]

    return AcceptanceResult(
        ranking=ranking[display_columns].copy(),
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


def get_secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


@st.cache_resource(show_spinner=False)
def get_supabase_client() -> Client:
    url = get_secret("SUPABASE_URL")
    key = get_secret("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError("尚未配置 Supabase。")
    return create_client(url, key)


@st.cache_data(ttl=60, show_spinner=False)
def load_employee_master() -> pd.DataFrame:
    client = get_supabase_client()
    records: list[dict[str, object]] = []
    start = 0
    page_size = 1000

    while True:
        response = (
            client.table(EMPLOYEE_TABLE)
            .select("record_key,user_id,erp,name,updated_at")
            .order("name")
            .range(start, start + page_size - 1)
            .execute()
        )
        page = response.data or []
        records.extend(page)
        if len(page) < page_size:
            break
        start += page_size

    columns = ["record_key", "user_id", "erp", "name", "updated_at"]
    return pd.DataFrame(records, columns=columns)


def refresh_employee_cache() -> None:
    load_employee_master.clear()


def upsert_employee_master(records_df: pd.DataFrame) -> int:
    if records_df.empty:
        return 0

    now = datetime.now(timezone.utc).isoformat()
    records = []
    for row in records_df.to_dict("records"):
        records.append(
            {
                "record_key": clean_name(row.get("record_key")),
                "user_id": clean_name(row.get("user_id")) or None,
                "erp": clean_name(row.get("erp")) or None,
                "name": clean_name(row.get("name")),
                "updated_at": now,
            }
        )

    get_supabase_client().table(EMPLOYEE_TABLE).upsert(
        records, on_conflict="record_key"
    ).execute()
    refresh_employee_cache()
    return len(records)


def delete_employee_records(record_keys: list[str]) -> int:
    keys = sorted({clean_name(value) for value in record_keys if clean_name(value)})
    if not keys:
        return 0
    get_supabase_client().table(EMPLOYEE_TABLE).delete().in_(
        "record_key", keys
    ).execute()
    refresh_employee_cache()
    return len(keys)


def make_excel_output(result: AcceptanceResult, gap_minutes: int) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        result.ranking.to_excel(writer, sheet_name="验收人效排名", index=False)

        method = pd.DataFrame(
            {
                "项目": [
                    "统计人员",
                    "验收量",
                    "有效工作时长",
                    "连续工作间隔",
                    "小时人效",
                    "人员匹配",
                    "总有效工作时长",
                ],
                "计算口径": [
                    "仅统计可在人员主数据中匹配到实际姓名的账号",
                    "按验收人汇总有效记录中的验收量",
                    "按验收人和日期合并重叠时间区间；相邻记录间隔在阈值内视为连续工作",
                    f"当前设置为 {gap_minutes} 分钟；超过该时间的空档不计入有效工作时长",
                    "验收量 ÷ 有效工作小时",
                    "先匹配用户编码/Use ID，再匹配ERP；账号中的@域名会自动移除",
                    "排名表中所有员工的有效工作时长相加",
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
        ranking_sheet.autofilter(0, 0, len(result.ranking), len(result.ranking.columns) - 1)
        ranking_sheet.set_row(0, 24, header_format)
        ranking_sheet.set_column("A:A", 8, integer_format)
        ranking_sheet.set_column("B:B", 24, text_format)
        ranking_sheet.set_column("C:C", 22, text_format)
        ranking_sheet.set_column("D:D", 14, integer_format)
        ranking_sheet.set_column("E:E", 20, text_format)
        ranking_sheet.set_column("F:F", 14, decimal_format)

        method_sheet = writer.sheets["计算口径"]
        method_sheet.set_row(0, 24, header_format)
        method_sheet.set_column("A:A", 20)
        method_sheet.set_column("B:B", 90)
        method_sheet.set_default_row(28)

    return output.getvalue()


def make_employee_master_excel(employee_df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    export_df = employee_df.rename(
        columns={"user_id": "用户编码", "erp": "ERP", "name": "姓名"}
    )[["用户编码", "ERP", "姓名"]]
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, sheet_name="人员主数据", index=False)
        worksheet = writer.sheets["人员主数据"]
        workbook = writer.book
        header_format = workbook.add_format(
            {
                "bold": True,
                "font_color": "#FFFFFF",
                "bg_color": "#1F4E78",
                "border": 1,
                "align": "center",
            }
        )
        worksheet.set_row(0, 24, header_format)
        worksheet.set_column("A:C", 24)
        worksheet.freeze_panes(1, 0)
    return output.getvalue()


def render_database_setup_message() -> None:
    st.error("人员主数据库尚未连接。请先完成 Supabase 设置。")
    with st.expander("查看需要填写的 Streamlit Secrets"):
        st.code(
            'SUPABASE_URL = "你的 Supabase Project URL"\n'
            'SUPABASE_KEY = "你的 Supabase service_role key"\n'
            'ADMIN_PASSWORD = "你自己设置的管理密码"',
            language="toml",
        )


def is_admin_authenticated() -> bool:
    expected_password = get_secret("ADMIN_PASSWORD")
    if not expected_password:
        st.error("尚未在 Streamlit Secrets 中设置 ADMIN_PASSWORD。")
        return False

    if st.session_state.get("employee_admin_authenticated", False):
        return True

    with st.form("employee_admin_login"):
        password = st.text_input("管理密码", type="password")
        submitted = st.form_submit_button("进入人员主数据管理", use_container_width=True)
        if submitted:
            if hmac.compare_digest(password, expected_password):
                st.session_state["employee_admin_authenticated"] = True
                st.rerun()
            else:
                st.error("管理密码不正确。")
    return False


def render_employee_management() -> None:
    st.subheader("人员主数据管理")
    st.caption("人员数据会保存在云端数据库中。上传一次后，验收及后续所有模块会自动使用。")

    try:
        employee_df = load_employee_master()
    except Exception as exc:
        render_database_setup_message()
        st.caption(f"连接详情：{exc}")
        return

    if not is_admin_authenticated():
        return

    top_left, top_right = st.columns([4, 1])
    top_left.metric("当前人员记录", f"{len(employee_df):,}")
    if top_right.button("退出管理", use_container_width=True):
        st.session_state["employee_admin_authenticated"] = False
        st.rerun()

    add_tab, bulk_tab, delete_tab = st.tabs(["添加或修改", "批量上传", "批量删除"])

    with add_tab:
        with st.form("add_employee_form", clear_on_submit=True):
            col1, col2, col3 = st.columns(3)
            user_id = col1.text_input("用户编码 / Use ID")
            erp = col2.text_input("ERP")
            name = col3.text_input("实际姓名 *")
            submitted = st.form_submit_button("保存人员", use_container_width=True)
            if submitted:
                user_id_clean = extract_operator_id(user_id)
                erp_clean = extract_operator_id(erp)
                name_clean = clean_name(name)
                record_key = build_record_key(user_id_clean, erp_clean)
                if not name_clean:
                    st.error("请输入实际姓名。")
                elif not record_key:
                    st.error("用户编码 / Use ID和ERP至少填写一项。")
                else:
                    record = pd.DataFrame(
                        [
                            {
                                "record_key": record_key,
                                "user_id": user_id_clean,
                                "erp": erp_clean,
                                "name": name_clean,
                            }
                        ]
                    )
                    upsert_employee_master(record)
                    st.success("人员信息已保存。相同用户编码再次保存时会更新原记录。")
                    st.rerun()

    with bulk_tab:
        st.write("上传 Excel 后，程序会按用户编码 / Use ID更新或新增人员。")
        bulk_file = st.file_uploader(
            "批量人员表",
            type=["xlsx", "xls"],
            key="employee_bulk_upload",
        )
        if bulk_file is not None:
            try:
                bulk_bytes = bulk_file.getvalue()
                sheets = get_sheet_names(bulk_bytes)
                sheet = st.selectbox("选择工作表", sheets, key="employee_bulk_sheet")
                raw_df = read_excel_sheet(bulk_bytes, sheet)
                import_df = prepare_employee_import(raw_df)
                preview = import_df.rename(
                    columns={"user_id": "用户编码", "erp": "ERP", "name": "姓名"}
                )[["用户编码", "ERP", "姓名"]]
                st.dataframe(preview.head(100), use_container_width=True, hide_index=True)
                st.caption(f"识别到 {len(import_df):,} 条可导入记录。")
                if st.button("确认批量写入", type="primary", use_container_width=True):
                    count = upsert_employee_master(import_df)
                    st.success(f"已写入 {count:,} 条人员记录。")
                    st.rerun()
            except Exception as exc:
                st.error(f"批量人员表处理失败：{exc}")

    with delete_tab:
        if employee_df.empty:
            st.info("当前没有人员记录。")
        else:
            editable = employee_df.copy()
            editable.insert(0, "选择删除", False)
            edited = st.data_editor(
                editable,
                use_container_width=True,
                hide_index=True,
                disabled=["record_key", "user_id", "erp", "name", "updated_at"],
                column_order=["选择删除", "user_id", "erp", "name", "updated_at"],
                column_config={
                    "选择删除": st.column_config.CheckboxColumn("选择删除"),
                    "user_id": "用户编码 / Use ID",
                    "erp": "ERP",
                    "name": "实际姓名",
                    "updated_at": "最后更新时间",
                },
                key="employee_delete_editor",
            )
            selected_keys = edited.loc[
                edited["选择删除"].fillna(False), "record_key"
            ].tolist()
            st.caption(f"已选择 {len(selected_keys):,} 条记录。")
            confirm_delete = st.checkbox(
                "我确认删除所选人员",
                key="confirm_employee_delete",
            )
            if st.button(
                "删除所选人员",
                disabled=not selected_keys or not confirm_delete,
                use_container_width=True,
            ):
                count = delete_employee_records(selected_keys)
                st.success(f"已删除 {count:,} 条人员记录。")
                st.rerun()

            st.divider()
            st.markdown("**通过 Excel 批量删除**")
            st.caption("删除表只需包含用户编码 / Use ID或ERP列。")
            delete_file = st.file_uploader(
                "批量删除名单",
                type=["xlsx", "xls"],
                key="employee_bulk_delete",
            )
            if delete_file is not None:
                try:
                    delete_bytes = delete_file.getvalue()
                    delete_sheets = get_sheet_names(delete_bytes)
                    delete_sheet = st.selectbox(
                        "删除名单工作表",
                        delete_sheets,
                        key="employee_delete_sheet",
                    )
                    delete_df = read_excel_sheet(delete_bytes, delete_sheet)
                    user_id_col = find_column(
                        delete_df.columns, EMPLOYEE_COLUMN_ALIASES["user_id"]
                    )
                    erp_col = find_column(delete_df.columns, EMPLOYEE_COLUMN_ALIASES["erp"])
                    if user_id_col is None and erp_col is None:
                        raise ValueError("删除名单中未找到用户编码 / Use ID或ERP列。")

                    target_ids = set()
                    target_erps = set()
                    if user_id_col:
                        target_ids = {
                            compact_key(extract_operator_id(value))
                            for value in delete_df[user_id_col].tolist()
                            if compact_key(extract_operator_id(value))
                        }
                    if erp_col:
                        target_erps = {
                            compact_key(extract_operator_id(value))
                            for value in delete_df[erp_col].tolist()
                            if compact_key(extract_operator_id(value))
                        }

                    matched_keys = employee_df.loc[
                        employee_df["user_id"].map(compact_key).isin(target_ids)
                        | employee_df["erp"].map(compact_key).isin(target_erps),
                        "record_key",
                    ].tolist()
                    st.info(f"将匹配删除 {len(matched_keys):,} 条现有人员记录。")
                    confirm_bulk_delete = st.checkbox(
                        "我确认按上传名单批量删除",
                        key="confirm_bulk_employee_delete",
                    )
                    if st.button(
                        "执行批量删除",
                        disabled=not matched_keys or not confirm_bulk_delete,
                        type="primary",
                        use_container_width=True,
                    ):
                        count = delete_employee_records(matched_keys)
                        st.success(f"已批量删除 {count:,} 条人员记录。")
                        st.rerun()
                except Exception as exc:
                    st.error(f"批量删除名单处理失败：{exc}")

    st.divider()
    st.subheader("当前人员主数据")
    display_df = employee_df.rename(
        columns={
            "user_id": "用户编码 / Use ID",
            "erp": "ERP",
            "name": "实际姓名",
            "updated_at": "最后更新时间",
        }
    )[["用户编码 / Use ID", "ERP", "实际姓名", "最后更新时间"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.download_button(
        "下载当前人员主数据",
        data=make_employee_master_excel(employee_df),
        file_name="人员主数据.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def render_acceptance_analysis() -> None:
    try:
        employee_df = load_employee_master()
        if employee_df.empty:
            st.warning("人员主数据为空，请先进入“人员主数据管理”批量上传人员表。")
            return
        employee_lookup = build_employee_lookup(employee_df)
    except Exception as exc:
        render_database_setup_message()
        st.caption(f"连接详情：{exc}")
        return

    with st.sidebar:
        st.header("验收数据")
        acceptance_file = st.file_uploader(
            "上传验收明细表",
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

    st.caption(f"人员主数据已自动载入：{len(employee_df):,} 条，无需重复上传。")

    if acceptance_file is None:
        st.info("请在左侧上传验收明细表。")
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

    if not result.invalid_rows.empty:
        st.warning(
            f"共有 {len(result.invalid_rows):,} 行因账号、验收量或时间无效而未计入人效。"
        )

    st.caption(
        f"计入排名的有效记录：{result.valid_row_count:,} 行。"
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
- **统计人员**：仅统计可在人员主数据中匹配到实际姓名的账号。
- **验收量**：按验收人汇总有效记录中的验收量。
- **验收人账号**：自动去除括号和邮箱域名，例如 `US018958(US018958@jd.com)` 转为 `US018958`。
- **实际姓名**：先匹配人员主数据中的用户编码 / Use ID，再匹配 ERP。
- **有效工作时长**：按验收人和日期合并重叠时间；相邻记录间隔不超过 **{int(gap_minutes)} 分钟**时视为连续工作。
- **小时人效**：验收量 ÷ 有效工作小时。
- **总有效工作时长**：排名表中所有员工的有效工作时长相加。
"""
        )


def render_app() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
    st.title(APP_TITLE)
    st.caption("验收小时人效｜统一人员主数据｜后续扩展上架、复核、打包和拣货")

    with st.sidebar:
        page = st.radio(
            "功能",
            ["验收人效分析", "人员主数据管理"],
            key="app_page",
        )
        st.divider()
        st.caption("人员主数据保存在云端，所有业务模块统一匹配实际姓名。")

    if page == "人员主数据管理":
        render_employee_management()
    else:
        render_acceptance_analysis()


if __name__ == "__main__":
    render_app()
