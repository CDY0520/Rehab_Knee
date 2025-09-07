"""
파일명: app/pages/04_운동일기.py
설명:
  - 환자가 앞서 처방받은 내용(JSON)을 불러와 주간 스케줄을 직접 선택하고,
    날짜별 운동일기를 작성·저장한다.
  - 처방의 주당 횟수에 맞춰 요일을 선택하면 '이번 주 계획표'를 생성한다.
  - 운동 전 문진표, 운동종류, 수행시간(분)을 일자별로 기록한다.
  - 저장 시 JSON/TXT로 내려받을 수 있으며, 이후 모니터링 리포트 페이지에서 재사용 가능.

블록 구성:
  0) 임포트/유틸
  1) 처방 불러오기(업로드 또는 자동 최신)
  2) 주간 계획: 기준주 선택 + 요일 선택(주당 횟수 가이드)
  3) 운동일기 입력: 문진표 + 운동종류 + 수행시간(분)
  4) 저장: JSON/TXT 다운로드

사용 예:
  streamlit run app/pages/04_운동일기.py
"""
from __future__ import annotations
from pathlib import Path
import json
import glob
import datetime as dt
import io
import streamlit as st

# ─────────────────────────────────────────────────────
# 0) 임포트/유틸
# ─────────────────────────────────────────────────────
st.markdown("<h2 style='text-align:center;'>운동일기</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "출처: 국립재활원「뇌졸중 장애인의 건강생활 가이드」"
    "</p>",
    unsafe_allow_html=True,
)

def latest_prescription() -> str | None:
    paths = sorted(
        glob.glob("*.json") + glob.glob("results/**/*.json", recursive=True),
        key=lambda p: Path(p).stat().st_mtime,
        reverse=True,
    )
    # 간단 휴리스틱: 파일명에 '_홈운동처방' 포함 우선
    for p in paths:
        if "_홈운동처방" in Path(p).stem:
            return p
    return paths[0] if paths else None

def weekday_kor(idx: int) -> str:
    return ["월","화","수","목","금","토","일"][idx]

def monday_of(date: dt.date) -> dt.date:
    return date - dt.timedelta(days=date.weekday())

def date_label(d: dt.date) -> str:
    return f"{d.strftime('%Y-%m-%d')}({weekday_kor(d.weekday())})"

# state
if "diary" not in st.session_state:
    st.session_state.diary = {}   # {"YYYY-MM-DD": {...}}

# ─────────────────────────────────────────────────────
# 1) 처방 불러오기
# ─────────────────────────────────────────────────────
st.subheader("1) 처방 불러오기")
col_u1, col_u2 = st.columns([2,1])
uploaded = col_u1.file_uploader("03_운동처방에서 저장한 JSON 업로드", type=["json"])
auto_path = latest_prescription()
col_u2.write("최근 파일 자동 탐색")
col_u2.code(auto_path or "(없음)", language="text")

presc_data = None
if uploaded is not None:
    presc_data = json.loads(uploaded.read().decode("utf-8"))
elif auto_path:
    with open(auto_path, "r", encoding="utf-8") as f:
        presc_data = json.load(f)

if not presc_data:
    st.info("처방 JSON을 업로드하거나, 상단 자동 탐색 경로에 파일이 있도록 저장하세요.")
    st.stop()

session_id = presc_data.get("session_id", "")
patient = presc_data.get("patient", {})
name = patient.get("name", "")
goal = patient.get("goal", "")
presc = presc_data.get("prescription", {})

# 처방에서 사용 중인 운동명 수집
exercise_names = []
for k, v in presc.items():
    if v and v.get("use"):
        exercise_names.append(v.get("title", k))
exercise_names = sorted(set(exercise_names)) or ["자유선택"]

st.success(f"세션: {session_id} | 환자: {name} | 목표: {goal}")
st.caption("불러온 처방의 운동 항목을 운동일기에서 선택할 수 있습니다.")

# 주당 권장 횟수 합(가이드)
weekly_target = sum(int(v.get("days_per_week",0)) for v in presc.values() if v.get("use"))
st.caption(f"※ 처방 기준 주당 총 횟수(참고): {weekly_target}회")

# ─────────────────────────────────────────────────────
# 2) 주간 계획
# ─────────────────────────────────────────────────────
st.subheader("2) 이번 주 계획 세우기")
today = dt.date.today()
base_week = st.date_input("기준 날짜(해당 주에 대한 계획을 작성)", value=today)
mon = monday_of(base_week)
week_days = [mon + dt.timedelta(days=i) for i in range(7)]
labels = [date_label(d) for d in week_days]

# 요일 선택
plan_days = st.multiselect("운동할 요일 선택", labels, default=labels[:min(len(labels), max(weekly_target, 3))])

# ─────────────────────────────────────────────────────
# 3) 운동일기 입력
# ─────────────────────────────────────────────────────
st.subheader("3) 운동일기 작성")
st.caption("운동 전 문진표에서 하나라도 해당되면 그날 운동을 중지하고 휴식하세요.")

QUEST_ITEMS = [
    "몸이 나른하다", "열이 있다", "두통이 있다", "숨이 차다",
    "어지러움이 있다", "기침/가래가 나온다", "신체에 통증이 있다", "기타 이상",
    "밤에 잠을 잘 못 잤다", "가슴이나 배가 아프다", "식사나 수분 섭취가 부족했다",
]

def diary_block(date_str: str):
    """일자별 입력 블록"""
    d_state = st.session_state.diary.get(date_str, {
        "precheck": [],
        "exercises": [],  # [{"name":..., "minutes":...}]
        "memo": "",
    })

    with st.expander(f"{date_str} 기록하기", expanded=False):
        pre = st.multiselect("운동 전 문진표(해당되는 항목 선택)", QUEST_ITEMS, default=d_state["precheck"], key=f"pre_{date_str}")
        st.markdown("**운동기록**")
        # 간단 반복 입력 UI
        rows = []
        for i in range(1, 6):
            c1, c2 = st.columns([2,1])
            name_i = c1.selectbox(f"운동종류 {i}", [""] + exercise_names, index=(exercise_names.index(d_state["exercises"][i-1]["name"])+1 if i-1 < len(d_state["exercises"]) and d_state["exercises"][i-1]["name"] in exercise_names else 0), key=f"name_{date_str}_{i}")
            min_i  = c2.number_input(f"수행시간 {i}(분)", 0, 300, (d_state["exercises"][i-1]["minutes"] if i-1 < len(d_state["exercises"]) else 0), key=f"min_{date_str}_{i}")
            if name_i and min_i>0:
                rows.append({"name": name_i, "minutes": int(min_i)})

        memo = st.text_area("메모(선택)", value=d_state["memo"], key=f"memo_{date_str}")

        # 저장 버튼
        if st.button("이 날짜 저장", key=f"save_{date_str}"):
            st.session_state.diary[date_str] = {"precheck": pre, "exercises": rows, "memo": memo}
            st.success(f"{date_str} 저장 완료")

# 계획한 요일들만 입력 블록 제공
for lb in plan_days:
    diary_block(lb.split("(")[0])

# ─────────────────────────────────────────────────────
# 4) 저장
# ─────────────────────────────────────────────────────
st.markdown("---")
if st.button("운동일기 전체 저장/다운로드"):
    # 요약 텍스트
    lines = []
    lines.append(f"세션 ID: {session_id} | 환자: {name} | 목표: {goal}")
    lines.append(f"주간 계획: {', '.join(plan_days) if plan_days else '계획 없음'}")
    lines.append("")

    for k in sorted(st.session_state.diary.keys()):
        rec = st.session_state.diary[k]
        lines.append(f"[{k}]")
        if rec["precheck"]:
            lines.append("· 운동 전 문진표: " + ", ".join(rec["precheck"]))
        if rec["exercises"]:
            for ex in rec["exercises"]:
                lines.append(f"· 운동: {ex['name']} / {ex['minutes']}분")
        if rec["memo"]:
            lines.append("· 메모: " + rec["memo"])
        lines.append("")

    txt = "\n".join(lines) if lines else "기록 없음."
    json_payload = {
        "session_id": session_id,
        "patient": patient,
        "goal": goal,
        "week_monday": mon.strftime("%Y-%m-%d"),
        "plan_days": plan_days,
        "diary": st.session_state.diary,
    }

    base = f"{(session_id or 's000')}_{name or 'patient'}_{mon.strftime('%Y%m%d')}_운동일기"
    st.download_button("운동일기 다운로드(.txt)",
                       data=io.BytesIO(txt.encode("utf-8")).getvalue(),
                       file_name=f"{base}.txt", mime="text/plain")
    st.download_button("원본 데이터 저장(.json)",
                       data=io.BytesIO(json.dumps(json_payload, ensure_ascii=False, indent=2).encode("utf-8")).getvalue(),
                       file_name=f"{base}.json", mime="application/json")

# 안내(작성법 + 촬영 권고 간단 버전)
st.markdown("---")
st.subheader("운동일기 작성법")
st.markdown("""
1) 운동 전 문진표를 먼저 작성하세요. 이상이 있으면 그날 운동을 쉬세요.  
2) 운동기록에 **운동종류**와 **수행시간(분)** 을 적으세요.  
3) 작성한 일기는 다음 방문 때 제출하면 평가에 반영됩니다.
""")
st.caption("영상 촬영 권고: 측면·정면 중 과제에 맞는 각도로 촬영, 전신이 프레임에 들어오게, 흔들림 최소화.")
