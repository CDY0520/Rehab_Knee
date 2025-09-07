"""
파일명: app/pages/03_운동처방.py
설명:
  - 개인 홈운동 처방 입력 페이지.
  - 세션ID/환자정보/목표 입력 후 운동 항목(유산소/스트레칭/근력/균형/기타)을 작성.
  - 단계 사용 여부(준비운동/본운동/마무리운동) 체크:
      · 체크하지 않으면 -> '처방 운동'으로만 리포트 출력(단순 목록).
      · 체크하면 -> 선택된 단계 순서(준비→본→마무리)로 리포트 그룹핑. 각 운동은 단계 선택으로 배치.
  - 항목별 입력: 주당 횟수, 1세트 반복, 세트 수, 예상 시간(분), 세부 동작/준비물, 동영상 링크, 유의사항, 단계(미지정/준비/본/마무리).
  - 값이 0 또는 빈칸이면 해당 항목은 리포트에서 자동 생략.
  - 치료사용 리포트와 환자용 체크리스트를 TXT/JSON으로 다운로드.

블록 구성:
  0) 임포트 및 제목/출처
  1) 기본 정보 + 단계 사용 여부
  2) 운동 항목 입력(항목별 + 단계 배치)
  3) 리포트 생성(단계 미사용: 단순 / 단계 사용: 그룹핑)
  4) 저장(TXT/JSON)

사용 예:
  streamlit run app/pages/03_운동처방.py
"""
from pathlib import Path
import io
import json
import streamlit as st

# ─────────────────────────────────────────────────────────
# 0) 제목 + 출처(작은 글씨)
# ─────────────────────────────────────────────────────────
st.markdown("<h2 style='text-align:center;'>재활 홈운동 처방</h2>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-size:12px; color:gray;'>"
    "출처: 국립재활원 「뇌병변 장애인의 재활체육 지침서」, "
    "「뇌졸중 장애인의 건강생활 가이드」"
    "</p>",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────
# 1) 기본 정보 + 단계 사용 여부
# ─────────────────────────────────────────────────────────
st.subheader("환자 기본 정보")
c0, c1, c2, c3 = st.columns(4)
session_id = c0.text_input("세션 ID (예: s001)")
name = c1.text_input("이름")
age = c2.text_input("나이")
dx = c3.text_input("진단명/특이사항")
goal = st.text_area("활동 목표", placeholder="예) 보행 안정성 확보, 무릎 과신전 줄이기, 앉았다 일어서기 향상")

st.subheader("단계 사용 여부")
d1, d2, d3 = st.columns(3)
use_warmup   = d1.checkbox("준비운동 사용", value=False)
use_main     = d2.checkbox("본운동 사용", value=False)
use_cooldown = d3.checkbox("마무리운동 사용", value=False)
any_phase = use_warmup or use_main or use_cooldown

# ─────────────────────────────────────────────────────────
# 2) 운동 항목 입력(항목별 + 단계 배치)
# ─────────────────────────────────────────────────────────
PHASE_OPTIONS = ["미지정", "준비운동", "본운동", "마무리운동"]

def block(title: str, key_prefix: str, allow_custom_name: bool = False):
    with st.expander(title, expanded=True):
        use = st.checkbox(f"{title} 포함", value=True, key=f"use_{key_prefix}")
        custom_name = ""
        if allow_custom_name:
            custom_name = st.text_input("기타 운동명", key=f"custom_{key_prefix}")

        cc1, cc2, cc3, cc4 = st.columns(4)
        days = cc1.number_input("주 당 횟수(회)", 0, 14, 0, key=f"days_{key_prefix}")
        reps = cc2.number_input("1세트 몇 번(회)", 0, 200, 0, key=f"reps_{key_prefix}")
        sets = cc3.number_input("총 몇 세트(세트)", 0, 50, 0, key=f"sets_{key_prefix}")
        mins = cc4.number_input("예상 소요 시간(분)", 0, 300, 0, key=f"mins_{key_prefix}")

        detail = st.text_area("세부 동작/준비물", key=f"detail_{key_prefix}",
                              placeholder="예) 빠르게 걷기 / 밴드 레드 3세트")
        video  = st.text_input("동영상 링크(선택)", key=f"video_{key_prefix}", placeholder="https://")
        caution = st.text_area("유의사항", key=f"caution_{key_prefix}",
                               placeholder="예) 통증 시 중단, 안전 보조자 동반 등")

        phase_col = st.columns(1)[0]
        phase = phase_col.selectbox("단계 배치", PHASE_OPTIONS, index=0, key=f"phase_{key_prefix}")
    return {
        "use": use,
        "title": (custom_name.strip() if allow_custom_name and custom_name.strip() else title),
        "days_per_week": int(days),
        "reps_per_set": int(reps),
        "sets_total": int(sets),
        "minutes_expected": int(mins),
        "detail": (detail or "").strip(),
        "video": (video or "").strip(),
        "caution": (caution or "").strip(),
        "phase": phase,
    }

st.subheader("운동 항목 선택 및 처방")
presc = {
    "유산소":   block("유산소",   "aer"),
    "스트레칭": block("스트레칭", "str"),
    "근력":     block("근력",     "res"),
    "균형":     block("균형",     "bal"),
    "기타":     block("기타",     "etc", allow_custom_name=True),
}

# ─────────────────────────────────────────────────────────
# 3) 리포트 생성
# ─────────────────────────────────────────────────────────
def nonzero(label, val, unit):
    return f"{label} {val}{unit}" if isinstance(val, int) and val > 0 else None

def compose_section_lines(title: str, v: dict) -> list[str]:
    """0/빈칸 자동 생략 규칙으로 한 운동 섹션을 텍스트 라인으로 변환."""
    # 모든 값이 0/빈칸이면 출력 생략
    if (
        v["days_per_week"] == 0
        and v["reps_per_set"] == 0
        and v["sets_total"] == 0
        and v["minutes_expected"] == 0
        and not v["detail"]
        and not v["video"]
        and not v["caution"]
    ):
        return []
    lines = [f"■ {title}"]
    row = []
    for seg in [
        nonzero("주", v["days_per_week"], "회"),
        nonzero("1세트", v["reps_per_set"], "회"),
        nonzero("총", v["sets_total"], "세트"),
        nonzero("", v["minutes_expected"], "분"),
    ]:
        if seg:
            row.append(seg.strip())
    if row:
        lines.append("- " + ", ".join(row))
    if v["detail"]:
        lines.append(f"- 세부: {v['detail']}")
    if v["video"]:
        lines.append(f"- 동영상: {v['video']}")
    if v["caution"]:
        lines.append(f"- 유의: {v['caution']}")
    lines.append("")
    return lines

def compose_checklist_lines(title: str, v: dict) -> list[str]:
    """환자용 체크리스트 라인 생성."""
    if (
        v["days_per_week"] == 0
        and v["reps_per_set"] == 0
        and v["sets_total"] == 0
        and v["minutes_expected"] == 0
        and not v["detail"]
    ):
        return []
    segs = []
    for seg in [
        nonzero("주", v["days_per_week"], "회"),
        nonzero("1세트", v["reps_per_set"], "회"),
        nonzero("총", v["sets_total"], "세트"),
        nonzero("", v["minutes_expected"], "분"),
    ]:
        if seg:
            segs.append(seg.strip())
    head = f"[ ] {title}: " + (", ".join(segs) if segs else "")
    lines = [head]
    if v["detail"]:
        lines.append(f"    · 동작: {v['detail']}")
    if v["video"]:
        lines.append(f"    · 동영상: {v['video']}")
    return lines

if st.button("리포트 생성"):
    # 공통 헤더
    head = []
    head.append("■ 환자 정보")
    head.append(f"- 세션 ID: {session_id}")
    head.append(f"- 이름: {name} | 나이: {age} | 진단: {dx}")
    head.append(f"- 목표: {goal}\n")

    # 단계 사용 여부에 따른 본문 생성
    lines = head.copy()
    checklist = [f"■ 홈운동 체크리스트", f"- 세션 ID: {session_id} | 이름: {name} | 목표: {goal}\n"]

    if not any_phase:
        # 단계 미사용: 단순 '처방 운동' 묶음
        body = []
        plist = []
        for k in ["유산소", "스트레칭", "근력", "균형", "기타"]:
            v = presc[k]
            if not v["use"]:
                continue
            body += compose_section_lines(v["title"], v)
            plist += compose_checklist_lines(v["title"], v)
        if body:
            lines.append("■ 처방 운동")
            lines += body
        if plist:
            checklist += plist
    else:
        # 단계 사용: 선택된 단계만 순서대로 출력, 각 운동은 phase 할당에 따라 배치
        def collect_for_phase(phase_name: str):
            body = []
            plist = []
            for k in ["유산소", "스트레칭", "근력", "균형", "기타"]:
                v = presc[k]
                if not v["use"]:
                    continue
                if v["phase"] != phase_name:
                    continue
                body += compose_section_lines(v["title"], v)
                plist += compose_checklist_lines(v["title"], v)
            return body, plist

        if use_warmup:
            warm_body, warm_pl = collect_for_phase("준비운동")
            if warm_body:
                lines.append("■ 준비운동")
                lines += warm_body
                checklist += warm_pl
        if use_main:
            main_body, main_pl = collect_for_phase("본운동")
            if main_body:
                lines.append("■ 본운동")
                lines += main_body
                checklist += main_pl
        if use_cooldown:
            cool_body, cool_pl = collect_for_phase("마무리운동")
            if cool_body:
                lines.append("■ 마무리운동")
                lines += cool_body
                checklist += cool_pl

        # 미지정 항목이 있고 단계 사용 중이면, 마지막에 '미지정'으로 묶어 제공
        misc_body, misc_pl = collect_for_phase("미지정")
        if misc_body:
            lines.append("■ 미지정(단계 선택 필요)")
            lines += misc_body
            checklist += misc_pl

    # 최종 텍스트
    report_txt = "\n".join(lines) if len(lines) > 0 else "입력된 처방이 없습니다."
    patient_txt = "\n".join(checklist + ["", "체크 방법: 수행했으면 [ ] → [✔]로 표시"])

    # 화면 출력
    st.subheader("처방 리포트")
    st.text(report_txt)

    # ─────────────────────────────────────────────────────
    # 4) 저장(TXT/JSON)
    # ─────────────────────────────────────────────────────
    base = f"{(session_id or 's000')}_{(name or 'patient')}_홈운동처방"
    st.download_button(
        "처방 리포트 다운로드(.txt)",
        data=io.BytesIO(report_txt.encode("utf-8")).getvalue(),
        file_name=f"{base}_report.txt",
        mime="text/plain",
    )
    payload = {
        "session_id": session_id,
        "patient": {"name": name, "age": age, "diagnosis": dx, "goal": goal},
        "prescription": presc,
        "use_phase": {"warmup": use_warmup, "main": use_main, "cooldown": use_cooldown},
    }
    st.download_button(
        "원본 데이터 저장(.json)",
        data=io.BytesIO(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")).getvalue(),
        file_name=f"{base}.json",
        mime="application/json",
    )
