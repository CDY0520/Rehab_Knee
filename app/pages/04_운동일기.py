# app/pages/04_운동일기.py
# -*- coding: utf-8 -*-
"""
페이지명: 주간 홈운동 계획 & 운동일기

설명:
 - results/json의 홈운동처방 JSON을 읽어 최근 처방을 요약하고 선택 요일로 월간 계획(⚪) 생성
 - 달력에서 날짜 클릭 → 일기 작성(제목=JSON title, 세트=sets_total, 반복=reps_per_set)
 - 운동 전 문진표에서 이상 선택 시 경보(⚠️) 기록. 달력 표시는 ⚠️ > ✅ > ⚪ > ❌
 - 일기 저장 후 선택 삭제 가능

상태 아이콘:
 - ⚠️ 경보  ✅ 완료  ⚪ 예정  ❌ 미작성

주요 기능:
 - 처방 자동 로드(세션 → results/json 내 홈운동처방*.json → 폴백)
 - [운동 처방 내용] 요약
 - [주간 운동 요일 선택] → 월간 계획 생성(해당 월 덮어쓰기)
 - 달력 클릭 → 일기 작성(운동 전 문진표 드롭다운 포함)
 - CSV 다운로드, 선택 삭제
 - 페이지 마지막에 “운동일기 작성법”

블록 구성:
 0) 경로/유틸
 1) 처방 로더(세션/JSON/폴백 + 정규화)
 2) 페이지 세팅 + [운동 처방 내용]
 3) [주간 운동 요일 선택] → 월간 스케줄 생성
 4) 달력 그리드(상태 아이콘)
 5) 일기 작성 폼(운동 전 문진표 + 세트/반복/완료/메모 + 선택 삭제)
 6) 다운로드
 7) (페이지 마지막) 운동일기 작성법
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0) 경로/유틸
# ──────────────────────────────────────────────────────────────────────────────
import os, io, re, glob, json, calendar
from datetime import date, datetime, timedelta
import pandas as pd
import streamlit as st

DATA_DIR  = "data"
PRESC_DIR = os.path.join("results", "json")   # 처방 JSON 탐색 디렉토리
DIARY_DIR = os.path.join(DATA_DIR, "diary")
SCHED_DIR = os.path.join(DATA_DIR, "schedule")
os.makedirs(DIARY_DIR, exist_ok=True)
os.makedirs(SCHED_DIR, exist_ok=True)

def diary_path(pid): return os.path.join(DIARY_DIR, f"{pid}_diary.csv")
def sched_path(pid): return os.path.join(SCHED_DIR, f"{pid}_schedule.csv")

def save_month_schedule(pid, first_day:date, last_day:date, rows:list):
    """해당 월 범위 기존 계획 제거 후 새 계획으로 덮어쓰기"""
    path = sched_path(pid)
    if os.path.exists(path):
        old = pd.read_csv(path)
        mask_other = ~(
            (old["patient_id"]==pid)
            & (pd.to_datetime(old["date"])>=pd.to_datetime(first_day))
            & (pd.to_datetime(old["date"])<=pd.to_datetime(last_day))
        )
        base = old[mask_other]
    else:
        base = pd.DataFrame()
    new_df = pd.DataFrame(rows)
    merged = pd.concat([base, new_df], ignore_index=True)
    merged = merged.sort_values(["patient_id","date","exercise_id"]).drop_duplicates(["patient_id","date","exercise_id"], keep="last")
    merged.to_csv(path, index=False)
    return merged

def render_delete_table(pid: str, target_dt: str, key_suffix: str = "main"):
    """해당 날짜 일기 표 + 체크 삭제 (폼 미사용, key_suffix로 고유 키 보장)"""
    if not os.path.exists(diary_path(pid)):
        st.caption("저장된 일기가 없습니다."); return

    df_all = pd.read_csv(diary_path(pid))
    df_day = df_all[(df_all["patient_id"]==pid) & (df_all["date"]==target_dt)].copy()
    if df_day.empty:
        st.caption("해당 날짜 저장된 일기가 없습니다."); return

    if "precheck_alert" in df_day.columns:
        df_day["precheck"] = df_day.apply(
            lambda r: f"⚠️ {r.get('precheck_item','')}" if bool(r.get("precheck_alert", False)) else "",
            axis=1
        )
    else:
        df_day["precheck"] = ""

    view = df_day[["patient_id","date","weekday","exercise_name","sets","reps","done","precheck","exercise_id"]].copy()
    view.insert(0, "삭제", False)
    view["__key"] = view["date"].astype(str) + "||" + view["exercise_id"].astype(str)

    editor_key = f"day_editor_{target_dt}_{key_suffix}"
    edited = st.data_editor(
        view.drop(columns=["exercise_id"]),
        hide_index=True,
        disabled=[c for c in view.columns if c not in ("삭제","__key")],
        use_container_width=True,
        key=editor_key,
    )

    to_del_keys = edited.loc[edited["삭제"]==True, "__key"].tolist()
    if st.button("선택 삭제", type="secondary", key=f"del_btn_{target_dt}_{key_suffix}"):
        if not to_del_keys:
            st.info("삭제할 항목을 선택하세요."); return
        df_all["__key"] = df_all["date"].astype(str) + "||" + df_all["exercise_id"].astype(str)
        remain = df_all[~df_all["__key"].isin(to_del_keys)].drop(columns="__key")
        remain.to_csv(diary_path(pid), index=False)
        st.success("선택 항목을 삭제했습니다.")
        st.dataframe(
            remain[(remain["patient_id"]==pid) & (remain["date"]==target_dt)]
                  [["patient_id","date","weekday","exercise_name","sets","reps","done"]],
            use_container_width=True
        )

# ──────────────────────────────────────────────────────────────────────────────
# 1) 처방 로더(세션/JSON/폴백 + 정규화)
# ──────────────────────────────────────────────────────────────────────────────
# 정규화(운동명=title 강제)
def _normalize_presc_records(records, pid) -> pd.DataFrame:
    iter_items = records.values() if isinstance(records, dict) else records
    norm = []
    for r in iter_items:
        if not isinstance(r, dict):
            continue
        title = str(r.get("title") or "").strip()
        if not title:
            continue
        # 요청된 제외 규칙
        if title.replace(" ", "") in {"무릎신전등척성", "무릎신전등척성운동"}:
            continue

        norm.append({
            "patient_id": pid,
            "exercise_id": title,                 # id=title
            "exercise_name": title,               # 이름=title
            "days_per_week": int(r.get("days_per_week", r.get("weekly", 3)) or 0),
            "sets": int(r.get("sets_total", r.get("sets", 0)) or 0),
            "reps": int(r.get("reps_per_set", r.get("reps", 0)) or 0),
            "minutes_expected": int(r.get("minutes_expected", 0) or 0),
            "detail": r.get("detail", ""),
            "caution": r.get("caution", ""),
            "phase": r.get("phase", ""),
            "use": bool(r.get("use", True)),
        })
    df = pd.DataFrame(norm)
    if df.empty:
        return df
    # 사용 + 유효값만
    return df[(df["use"] == True) & ((df["sets"] > 0) | (df["reps"] > 0))].reset_index(drop=True)

def _find_prescription_file(pid: str) -> str | None:
    if not os.path.isdir(PRESC_DIR): return None
    files = [p for p in glob.glob(os.path.join(PRESC_DIR, "*.json")) if "qmetrics" not in os.path.basename(p)]
    if not files: return None
    pid_variants = {pid, pid.lower(), pid.upper()}
    m = re.search(r"\d+", pid)
    if m:
        n = int(m.group()); pid_variants |= {f"s{n:03d}", f"S{n:03d}"}
    def rank(p:str):
        name = os.path.basename(p)
        has_kw  = ("홈운동처방" in name) or ("prescription" in name.lower())
        has_pid = any(v in name for v in pid_variants)
        return (int(has_pid and has_kw), int(has_kw), os.path.getmtime(p))
    files.sort(key=rank, reverse=True)
    return files[0]

# JSON 로더
def _load_presc_from_json(pid:str) -> pd.DataFrame:
    fp = _find_prescription_file(pid)
    if not fp:
        return pd.DataFrame()
    try:
        with open(fp, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # ✅ 'prescription'과 'prescriptions' 모두 지원
        if isinstance(payload, dict):
            if "prescription" in payload:      # ← 당신 파일 구조
                records = payload["prescription"]   # dict of {카테고리: {...}}
            elif "prescriptions" in payload:
                records = payload["prescriptions"]
            else:
                records = payload
        else:
            records = payload

        return _normalize_presc_records(records, pid)
    except Exception:
        return pd.DataFrame()

def load_latest_prescription(pid:str) -> pd.DataFrame:
    if "prescription_df" in st.session_state:
        df = st.session_state["prescription_df"].copy()
        df = df[df["patient_id"] == pid]
        if not df.empty: return df.reset_index(drop=True)
    dfj = _load_presc_from_json(pid)
    if not dfj.empty: return dfj
    return pd.DataFrame([{
        "patient_id": pid, "exercise_id": "보조 스쿼트", "exercise_name": "보조 스쿼트",
        "days_per_week": 3, "sets": 2, "reps": 12, "use": True
    }])

# ──────────────────────────────────────────────────────────────────────────────
# 2) 페이지 세팅 + [운동 처방 내용]
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="주간 홈운동 계획", layout="wide")
pid = st.session_state.get("patient_id", "P001")

st.title("주간 홈운동 계획")
st.markdown(
    "<p style='text-align:left; font-size:12px; color:gray;'>"
    "출처: 국립재활원「뇌졸중 장애인의 건강생활 가이드」"
    "</p>",
    unsafe_allow_html=True,
)

presc = load_latest_prescription(pid)
ex_list = presc["exercise_name"].tolist() if not presc.empty else []
days_per_week = int(presc["days_per_week"].max()) if not presc.empty else 3

st.subheader("운동 처방 내용")
if presc.empty:
    st.caption("처방이 없습니다.")
else:
    summary = " · ".join([f"{r.exercise_name}({int(r.sets)}세트×{int(r.reps)}회)"
                           for r in presc.drop_duplicates(subset=["exercise_id"]).itertuples()])
    st.markdown(f"- 주 {days_per_week}회 권장\n- 총 {len(ex_list)}개: {summary}")

# ──────────────────────────────────────────────────────────────────────────────
# 3) [주간 운동 요일 선택] → 월간 스케줄 생성
# ──────────────────────────────────────────────────────────────────────────────
weekday_labels = ["월","화","수","목","금","토","일"]
st.subheader("주간 운동 요일 선택")
cols = st.columns(7)
selected = []
default_idxs = [0,2,4] if days_per_week >= 3 else list(range(days_per_week))
for i, lab in enumerate(weekday_labels):
    with cols[i]:
        if st.checkbox(lab, value=(i in default_idxs), key=f"wk_{i}"):
            selected.append(i)
if len(selected) > days_per_week:
    selected = selected[:days_per_week]
    st.info(f"주 {days_per_week}회 기준. 처음 {days_per_week}개 요일만 반영합니다.")

st.subheader("달력")
st.markdown("상태: ⚠️ 경보 · ✅ 완료 · ⚪ 예정 · ❌ 미작성")

month_anchor = st.date_input("기준 날짜", value=date.today(), format="YYYY-MM-DD")
first_day = date(month_anchor.year, month_anchor.month, 1)
next_month_first = (first_day.replace(day=28) + timedelta(days=7)).replace(day=1)
last_day = next_month_first - timedelta(days=1)

if selected and not presc.empty:
    rows = []
    for d in pd.date_range(first_day, last_day, freq="D"):
        if d.weekday() in selected and d.date() >= date.today():
            for _, ex in presc.iterrows():
                rows.append({
                    "patient_id": pid,
                    "date": d.date().isoformat(),
                    "weekday": weekday_labels[d.weekday()],
                    "exercise_id": ex["exercise_id"],
                    "exercise_name": ex["exercise_name"],
                    "sets": int(ex["sets"]),
                    "reps": int(ex["reps"]),
                    "plan": True
                })
    save_month_schedule(pid, first_day, last_day, rows)

# ──────────────────────────────────────────────────────────────────────────────
# 4) 달력 그리드(상태 아이콘: ⚠️ > ✅ > ⚪ > ❌)
# ──────────────────────────────────────────────────────────────────────────────
def draw_month_grid(target_month:date):
    cal = calendar.Calendar(firstweekday=0)
    weeks = cal.monthdayscalendar(target_month.year, target_month.month)
    plan_df  = pd.read_csv(sched_path(pid)) if os.path.exists(sched_path(pid)) else pd.DataFrame()
    diary_df = pd.read_csv(diary_path(pid)) if os.path.exists(diary_path(pid)) else pd.DataFrame()

    if not diary_df.empty:
        done_mask  = diary_df["done"] == True if "done" in diary_df.columns else pd.Series([False]*len(diary_df))
        alert_mask = diary_df["precheck_alert"] == True if "precheck_alert" in diary_df.columns else pd.Series([False]*len(diary_df))
        done_dates  = set(diary_df.loc[done_mask, "date"].astype(str).unique())
        alert_dates = set(diary_df.loc[alert_mask, "date"].astype(str).unique())
    else:
        done_dates, alert_dates = set(), set()

    plan_dates = set(plan_df["date"].unique()) if not plan_df.empty else set()

    st.write(f"**{target_month.year}년 {target_month.month}월**")
    header = st.columns(7)
    for i, lab in enumerate(["월","화","수","목","금","토","일"]):
        header[i].markdown(f"<div style='text-align:center'>{lab}</div>", unsafe_allow_html=True)

    for week in weeks:
        cols = st.columns(7)
        for idx, num in enumerate(week):
            with cols[idx]:
                if num == 0:
                    st.markdown("&nbsp;", unsafe_allow_html=True); continue
                d = date(target_month.year, target_month.month, num); d_iso = d.isoformat()

                if d_iso in alert_dates:       badge = "⚠️"
                elif d_iso in done_dates:       badge = "✅"
                elif (d_iso in plan_dates) and (d >= date.today()): badge = "⚪"
                elif (d_iso in plan_dates) and (d < date.today()):  badge = "❌"
                else: badge = ""

                if st.button(f"{num} {badge}", key=f"btn_{d_iso}", use_container_width=True):
                    st.session_state["diary_date"] = d_iso

draw_month_grid(first_day)

# ──────────────────────────────────────────────────────────────────────────────
# 5) 운동일기 작성(문진표 드롭박스 + 저장 후 표/선택삭제)
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("운동일기 작성")
target_dt = st.session_state.get("diary_date")
if not target_dt:
    st.caption("달력에서 날짜를 클릭하세요.")
else:
    st.info(f"작성 날짜: {target_dt}")

    # 계획 로드 or 프리필
    if os.path.exists(sched_path(pid)):
        full_sched = pd.read_csv(sched_path(pid))
        day_plan = full_sched[(full_sched["patient_id"]==pid) & (full_sched["date"]==target_dt)]
    else:
        day_plan = pd.DataFrame()
    if day_plan.empty and not presc.empty:
        base = presc.copy()
        base["date"] = target_dt
        base["weekday"] = ["월","화","수","목","금","토","일"][datetime.fromisoformat(target_dt).weekday()]
        day_plan = base[["date","weekday","exercise_id","exercise_name","sets","reps"]]

    with st.form("diary_form", clear_on_submit=False):
        # 문진표 드롭다운(기타 포함)
        st.markdown("**운동 전 문진표**  \n*아래 항목 중 어느 하나라도 이상이 있으면 오늘 운동을 중지합니다.*")
        q_options = [
            "해당사항 없음", "몸이 나른하다", "어지럼증이 있다", "밤에 잘 자지 못했다", "열이 있다",
            "기침이나 가래가 나온다", "가슴이나 배가 아프다", "두통이 있다", "신체에 통증이 있다",
            "설사나 심한 변비가 있다", "숨이 차다", "기타(직접 입력)"
        ]
        pre_choice = st.selectbox("해당 항목 선택", q_options, index=0, key="pre_choice")
        etc_txt = st.text_input("기타 내용", "", key="pre_choice_etc") if pre_choice=="기타(직접 입력)" else ""
        pre_item_text = etc_txt if pre_choice=="기타(직접 입력)" else pre_choice
        pre_alert = pre_item_text not in ("", "해당사항 없음")

        rows = []

        if pre_alert:
            # 이상 선택 시: 수행 내역 숨기고 경보만 저장
            st.warning("운동 전 문진표에서 이상 항목이 선택되었습니다. 오늘 운동을 중지하세요.")
            # 경보 전용 레코드 1건만 저장
            rows.append({
                "patient_id": pid, "date": target_dt,
                "weekday": ["월","화","수","목","금","토","일"][datetime.fromisoformat(target_dt).weekday()],
                "exercise_id": "__ALERT__", "exercise_name": "경보",
                "sets": 0, "reps": 0, "done": False,
                "comment": "",
                "precheck_alert": True, "precheck_item": pre_item_text
            })
            submit = st.form_submit_button("경보 저장")
        else:
            st.write("**오늘 수행 내역**")
            for idx, row in day_plan.reset_index(drop=True).iterrows():
                c1,c2,c3,c4,c5 = st.columns([4,1,1,1,3])
                with c1: st.text(row["exercise_name"])
                with c2: sets = st.number_input("세트", 0, 20, int(row.get("sets",0)), key=f"sets_{idx}")
                with c3: reps = st.number_input("반복", 0, 200, int(row.get("reps",0)), key=f"reps_{idx}")
                with c4: done = st.checkbox("완료", value=False, key=f"done_{idx}")
                with c5: note = st.text_input("메모", value="", key=f"note_{idx}")
                rows.append({
                    "patient_id": pid, "date": target_dt, "weekday": row.get("weekday",""),
                    "exercise_id": row["exercise_id"], "exercise_name": row["exercise_name"],
                    "sets": int(sets), "reps": int(reps), "done": bool(done),
                    "comment": note, "precheck_alert": False, "precheck_item": ""
                })
            submit = st.form_submit_button("저장")

    # 저장 처리는 폼 밖에서
    if target_dt and rows and submit:
        new_df = pd.DataFrame(rows)
        if os.path.exists(diary_path(pid)):
            old = pd.read_csv(diary_path(pid))
            out = pd.concat([old, new_df], ignore_index=True)
            out = out.sort_values(["patient_id","date","exercise_id"]).drop_duplicates(
                ["patient_id","date","exercise_id"], keep="last"
            )
        else:
            out = new_df
        out.to_csv(diary_path(pid), index=False)
        st.success("운동일기를 저장했습니다.")
        st.session_state["refresh_table"] = True

    # 페이지 하단에서도 항상 현재 기록 관리 가능
    st.markdown("**저장된 운동일기(선택 삭제 가능)**")
    render_delete_table(pid, target_dt, key_suffix="single")
    st.session_state["refresh_table"] = False

# ──────────────────────────────────────────────────────────────────────────────
# 6) 다운로드
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("다운로드")
if os.path.exists(diary_path(pid)):
    df = pd.read_csv(diary_path(pid))
    buf = io.BytesIO(); df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button("운동일기 CSV 다운로드", buf.getvalue(), file_name=f"{pid}_diary.csv", mime="text/csv")
else:
    st.caption("저장된 일기가 없습니다.")

# ──────────────────────────────────────────────────────────────────────────────
# 7) (페이지 마지막) 운동일기 작성법
# ──────────────────────────────────────────────────────────────────────────────
ex_names = " · ".join(ex_list) if ex_list else "처방된 운동 없음"
with st.expander("운동일기 작성법", expanded=True):
    st.markdown(
        "- 날짜별 처방 운동을 기준으로 완료 여부를 기록합니다.\n"
        "- 각 운동의 세트·반복은 기본값(처방)에서 실제 수행값으로 수정해 입력합니다.\n"
        "- 운동 전 문진표에서 ‘해당사항 없음’을 제외한 항목 선택 시 경보로 기록되고 달력에 ⚠️로 표시됩니다.\n"
        "- 필요 시 메모를 남기고 저장 후 CSV로 내려받아 공유합니다."
    )
