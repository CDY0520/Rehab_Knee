"""
파일명: app/pages/02_동작분석.py
설명:
  - 첫 화면: 분석 과제 선택(보행/STS) + 리포트 생성 버튼만 표시.
  - 리포트 생성 후: 결과 본문 출력 → 하단에 치료사 코멘트 입력/저장 → 코멘트가 본문에 추가 표기.
  - 저장: 리포트 생성 이후에만 다운로드 버튼 노출, 기본 형식은 텍스트(.txt) 단일.
  - 최신 npz 파일을 자동 선택하여 분석(events.py 사용).

블록 구성:
  0) 임포트 및 경로 설정
  1) 최신 npz 자동 선택
  2) 보행/STS 문구 변환 함수
  3) Streamlit UI 흐름 제어(state): 과제 선택 → 리포트 생성 → 코멘트 저장 → TXT 다운로드

사용 예:
  streamlit run app/pages/02_동작분석.py
"""

import io
import json
import glob
from pathlib import Path

import numpy as np
import streamlit as st

# ── src/events 임포트 경로 설정 ─────────────────────────────────────────────
import sys
ROOT = Path(__file__).resolve().parents[2]   # .../Rehab_Knee
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import events  # src/events.py

# ── 최신 npz 자동 선택 ────────────────────────────────────────────────────
def latest_npz(dir_glob: str = "results/keypoints/*.npz") -> str | None:
    files = sorted(glob.glob(dir_glob), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return files[0] if files else None

# ── 보행 문구 변환(쉬운 표현 + 아이콘) ─────────────────────────────────────
def 문구_보행_측면(side_label: str, ev: dict, 무릎: dict, 선택: dict) -> list[str]:
    msgs = []
    hs_n = len(ev.get("HS_ms", []))
    to_n = len(ev.get("TO_ms", []))
    ms_n = len(ev.get("MS_ms", []))

    msgs.append(f"• {side_label} 발뒤꿈치 닿기: {'❌ 발생하지 않았습니다.' if hs_n == 0 else f' {hs_n}회 발생했습니다.'}")
    msgs.append(f"• {side_label} 발끝 차고 나가기: {'❌ 발생하지 않았습니다.' if to_n == 0 else f' {to_n}회 발생했습니다.'}")
    msgs.append(f"• {side_label} 중간 디딤(지지): {'❌ 확인되지 않았습니다.' if ms_n == 0 else f' {ms_n}회 확인되었습니다.'}")

    최대각 = 무릎.get("knee_max_deg", 0.0)
    과신전비율 = 무릎.get("hyperext_ratio_all", 0.0)
    if 과신전비율 > 0:
        msgs.append(f"• ⚠️ {side_label} 무릎: 뒤로 과하게 펴지는 현상(과신전)이 관찰됩니다. (최대 {최대각:.1f}도)")
    else:
        msgs.append(f"•  {side_label} 무릎: 과신전은 관찰되지 않았습니다. (최대 {최대각:.1f}도)")

    if 선택.get("stiff_knee_flag", False):
        msgs.append(f"• ⚠️ {side_label} 무릎: 다리를 앞으로 내딛을 때 무릎 굽힘이 부족하여 동작이 뻣뻣합니다.")
    else:
        msgs.append(f"•  {side_label} 무릎: 다리를 앞으로 내딛을 때 굽힘이 적절합니다.")

    tc_list = 선택.get("toe_clear_min_list")
    if tc_list:
        tc_mean = float(np.mean(tc_list))
        if tc_mean < 0.012:
            msgs.append(f"• 🔴 {side_label} 발: 다리를 앞으로 옮길 때 발이 충분히 들리지 않아 걸림 위험이 있습니다.")
        else:
            msgs.append(f"•  {side_label} 발: 다리를 앞으로 옮길 때 발 들림이 적절합니다.")
    return msgs

# ── STS 문구 변환(쉬운 표현 + 아이콘) ─────────────────────────────────────
def 문구_STS(ev: dict, m: dict) -> list[str]:
    msgs = []
    so_n = len(ev.get("seat_off_ms", []))
    fs_n = len(ev.get("full_stand_ms", []))
    cycles = m.get("cycles", 0)
    mean_sec = m.get("mean_cycle_sec", 0.0)

    if cycles == 0:
        msgs.append("• ❌ 앉았다 일어서는 동작이 탐지되지 않았습니다.")
        return msgs

    msgs.append(f"•  앉았다 일어서기 동작: 총 {cycles}회")
    msgs.append(f"• 평균 소요 시간: {mean_sec:.2f}초")
    if so_n == 0:
        msgs.append("• ⚠️ 엉덩이를 떼는 순간이 명확히 나타나지 않았습니다.")
    if fs_n == 0:
        msgs.append("• ⚠️ 완전히 일어선 상태가 명확히 나타나지 않았습니다.")
    return msgs

# ── UI 흐름 ────────────────────────────────────────────────────────────────
st.markdown("<h2 style='text-align:center;'>이벤트 기반 보행/STS 동작 분석 리포트</h2>", unsafe_allow_html=True)

# state 초기화
for k, v in {
    "report_task": "보행",
    "report_text": "",
    "npz_path": latest_npz(),
    "comment_text": "",
    "comment_applied": False,
    "report_ready": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ① 첫 화면: 과제 선택 + 리포트 생성
st.session_state.report_task = st.radio("분석 과제 선택", ["보행", "STS"], horizontal=True, index=0)

gen = st.button("리포트 생성")
if gen:
    if not st.session_state.npz_path:
        st.error("저장된 npz가 없습니다. 먼저 pose_probe.py로 결과를 생성하세요. (기본: results/keypoints/)")
        st.stop()

    npz_path = st.session_state.npz_path
    out_base = Path(npz_path).stem
    st.info(f"분석 대상 파일(자동 선택): **{Path(npz_path).name}**")

    # 분석 및 본문 생성
    if st.session_state.report_task == "보행":
        res = events.detect_events_bilateral(npz_path)
        lines = []
        for side_key, side_label in [("LEFT", "왼쪽"), ("RIGHT", "오른쪽")]:
            ev = res[side_key]["events"]
            무릎 = res[side_key]["metrics_knee_only"]
            선택 = res[side_key]["metrics_optional"]
            lines.append(f"■ {side_label} 다리")
            lines.extend(문구_보행_측면(side_label, ev, 무릎, 선택))
            lines.append("")
        st.session_state.report_text = "\n".join(lines)
        st.session_state.report_payload = {"task": "gait", "npz": npz_path, "result": res}
        st.session_state.out_name = f"{out_base}_report_gait.txt"
    else:
        res = events.detect_sts_events(npz_path)
        ev, m = res["events"], res["metrics"]
        lines = ["■ STS 분석"]
        lines.extend(문구_STS(ev, m))
        lines.append("")
        st.session_state.report_text = "\n".join(lines)
        st.session_state.report_payload = {"task": "sts", "npz": npz_path, "result": res}
        st.session_state.out_name = f"{out_base}_report_sts.txt"

    st.session_state.comment_text = ""
    st.session_state.comment_applied = False
    st.session_state.report_ready = True

# ② 리포트 표시
if st.session_state.report_ready and st.session_state.report_text:
    st.subheader("리포트")
    st.text(st.session_state.report_text)

    # ③ 코멘트 입력/저장(리포트 아래 표시)
    st.markdown("---")
    st.markdown("#### 치료사 코멘트")
    st.session_state.comment_text = st.text_area(
        "",
        value=st.session_state.comment_text,
        placeholder="환자분의 동작 특징, 주의사항, 연습 방법, 다음 단계 권고 등을 입력하세요.",
        height=120,
    )
    if st.button("코멘트 저장"):
        comment = st.session_state.comment_text.strip()
        if comment:
            appended = st.session_state.report_text + "\n" + "■ 치료사 코멘트\n" + "\n".join(
                f"• 🔴 {line.strip()}" for line in comment.splitlines() if line.strip()
            )
            st.session_state.report_text = appended
            st.session_state.comment_applied = True
            st.success("코멘트를 리포트에 추가했습니다.")
        else:
            st.warning("코멘트가 비어 있습니다.")

    # 코멘트가 반영된 최신 리포트 재표시
    if st.session_state.comment_applied:
        st.subheader("최종 동작 분석 리포트")
        st.text(st.session_state.report_text)

    # ④ 저장 버튼(리포트 생성 후에만 노출, 기본 TXT 단일)
    st.markdown("---")
    txt_buf = io.BytesIO(st.session_state.report_text.encode("utf-8"))
    st.download_button(
        "리포트 다운로드(.txt)",
        data=txt_buf.getvalue(),
        file_name=st.session_state.out_name,
        mime="text/plain",
    )
