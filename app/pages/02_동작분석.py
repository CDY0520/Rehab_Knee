"""
파일명: app/pages/02_동작분석.py
설명:
  - 좌측 사이드바에 탭 구성(파일 업로드 / 동작 분석 결과 / 최종 리포트)
  - 업로드한 npz 분석(events.py 활용), 선택된 측면만 결과 출력
  - 누락된 이벤트/지표는 "발생 없음"으로 표시
  - 동작 분석 결과에 치료사 코멘트 입력 → 최종 리포트에 반영
  - 한글 용어 + 영문 약어 병기
  - 위험 경고는 🔴, 주의 메시지는 ⚠️(노란색 세모 느낌표)

블록 구성:
  0) 임포트 및 경로 설정
  1) Streamlit 페이지 설정 + CSS 스타일 정의
  2) 유틸 함수 정의
  3) 사이드바 탭 메뉴 구성
  4) 탭1: 파일 업로드 및 리포트 생성
  5) 탭2: 동작 분석 결과(치료사 코멘트 입력)
  6) 탭3: 최종 리포트 표시 및 다운로드
"""

import io, sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ── src/events 경로 설정 ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
import events  # src/events.py

# ── 페이지 설정 / 스타일 ────────────────────────────────────────────────────
st.set_page_config(page_title="이벤트 기반 보행 동작 분석 리포트", layout="wide")
st.markdown("""
<style>
.main {background:#f7f8fb;}
.header{background:linear-gradient(135deg,#0f172a,#1e293b);color:#fff;
  padding:18px 20px;border-radius:16px;margin:10px 0 14px;font-weight:700;font-size:20px;}
.card{background:#fff;border:1px solid #e8eaf1;border-radius:16px;padding:16px;
  box-shadow:0 4px 14px rgba(15,23,42,.06);margin-bottom:14px;}
.h3{font-weight:700;font-size:16px;margin:0 0 10px 0;color:#0f172a}
.small{color:#64748b;font-size:12px}
.final{background:linear-gradient(180deg,#ffffff,#f1f5f9);
  border:1px solid #e2e8f0;border-radius:16px;padding:18px;}
</style>
""", unsafe_allow_html=True)
st.markdown("<div class='header'>이벤트 기반 보행 동작 분석 리포트</div>", unsafe_allow_html=True)

# ── 상태 초기화 ─────────────────────────────────────────────────────────────
if "report_ready" not in st.session_state:
    st.session_state.update({
        "npz_name":"", "npz_path":"", "side_key":None, "side_label":"",
        "rows":[], "report_text":"", "comment":"", "comment_applied":False,
        "out_name":"report_gait.txt", "payload":None, "report_ready":False
    })

# ── 유틸 ────────────────────────────────────────────────────────────────────
def infer_side_from_name(name: str) -> str | None:
    n = name.lower()
    if "left" in n or "_l" in n or "-l" in n:  return "LEFT"
    if "right" in n or "_r" in n or "-r" in n: return "RIGHT"
    return None

def pick_side_payload(res: dict, side_key: str):
    side = res.get(side_key, {}) or {}
    ev = side.get("events", {}) or {}
    m_k = (side.get("metrics_knee_only")
           or side.get("metrics_knee")
           or side.get("knee_metrics")
           or (side.get("metrics", {}) if isinstance(side.get("metrics"), dict) else {})
           or {})
    m_op = (side.get("metrics_optional")
            or side.get("optional_metrics")
            or {})
    return ev, m_k, m_op

def lines_to_df(lines: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"항목": lines})

# ── 결과 문구 생성 ──────────────────────────────────────────────────────────
def 문구_보행_측면(side_label: str, ev: dict, m_k: dict, m_op: dict) -> list[str]:
    msgs = []

    # 이벤트: HS/TO/MS
    hs_n = len(ev.get("HS_ms", []))
    to_n = len(ev.get("TO_ms", []))
    ms_n = len(ev.get("MS_ms", []))
    msgs.append(f"{side_label} 뒤꿈치 닿음(HS): {'발생 없음' if hs_n==0 else f'{hs_n}회 발생했습니다.'}")
    msgs.append(f"{side_label} 발끝 이탈(TO): {'발생 없음' if to_n==0 else f'{to_n}회 발생했습니다.'}")
    msgs.append(f"{side_label} 중간 디딤(MS): {'발생 없음' if ms_n==0 else f'{ms_n}회 확인되었습니다.'}")

    # 무릎 각도/과신전
    knee_max = float(m_k.get("knee_max_deg", np.nan)) if m_k else np.nan
    hyper_ratio = float(m_k.get("hyperext_ratio_all", 0.0)) if m_k else 0.0
    if np.isnan(knee_max):
        msgs.append(f"{side_label} 무릎 각도(Knee angle): 데이터 없음")
    else:
        if hyper_ratio > 0:
            msgs.append(f"⚠️ {side_label} 무릎: 과신전이 관찰됩니다. (최대 {knee_max:.1f}°)")
        else:
            msgs.append(f"{side_label} 무릎: 과신전은 관찰되지 않았습니다. (최대 {knee_max:.1f}°)")

    # 스윙 굴곡 부족
    if m_op.get("stiff_knee_flag", False):
        msgs.append(f"⚠️ {side_label} 무릎: 다리를 앞으로 내딛을 때 무릎 굽힘이 부족합니다.")
    else:
        msgs.append(f"{side_label} 무릎: 다리를 앞으로 내딛을 때 굽힘이 적절합니다.")

    return msgs

# ── 사이드바 네비 ──────────────────────────────────────────────────────────
with st.sidebar:
    nav = st.radio("메뉴", ["파일 업로드", "동작 분석 결과", "최종 리포트"], index=0)

# ── 1) 파일 업로드 탭 ───────────────────────────────────────────────────────
if nav == "파일 업로드":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>파일 업로드 (.npz)</div>", unsafe_allow_html=True)

    up = st.file_uploader("pose_probe 결과 .npz 파일을 업로드하세요", type=["npz"])
    side_choice = st.radio("측면", ["자동","왼쪽","오른쪽"], horizontal=True, index=0)
    gen = st.button("리포트 생성", use_container_width=True)

    if up:
        st.caption(f"선택: **{up.name}**")
    else:
        st.caption("선택된 파일이 없습니다.")

    if gen:
        if not up:
            st.error("npz 파일을 업로드하세요.")
            st.stop()
        # 업로드 파일 임시 저장
        tmp_dir = Path("results/tmp_upload"); tmp_dir.mkdir(parents=True, exist_ok=True)
        npz_path = str(tmp_dir / up.name)
        with open(npz_path, "wb") as f: f.write(up.getbuffer())

        # 이벤트 양측 분석 후 한쪽만 선택
        res = events.detect_events_bilateral(npz_path)
        if side_choice == "왼쪽":
            SIDE_KEY, SIDE_LABEL = "LEFT", "왼쪽"
        elif side_choice == "오른쪽":
            SIDE_KEY, SIDE_LABEL = "RIGHT", "오른쪽"
        else:
            inf = infer_side_from_name(up.name)
            SIDE_KEY = inf if inf else "LEFT"
            SIDE_LABEL = "왼쪽" if SIDE_KEY=="LEFT" else "오른쪽"

        ev, m_k, m_op = pick_side_payload(res, SIDE_KEY)
        one_side_lines = 문구_보행_측면(SIDE_LABEL, ev, m_k, m_op)

        # 상태 저장
        st.session_state.npz_name   = up.name
        st.session_state.npz_path   = npz_path
        st.session_state.side_key   = SIDE_KEY
        st.session_state.side_label = SIDE_LABEL
        st.session_state.rows       = one_side_lines
        st.session_state.report_text = f"■ {SIDE_LABEL} 다리\n" + "\n".join(one_side_lines)
        st.session_state.out_name   = f"{Path(up.name).stem}_report_gait_{SIDE_KEY.lower()}.txt"
        st.session_state.payload    = {"task":"gait","side":SIDE_KEY,"npz":npz_path}
        st.session_state.report_ready = True
        st.session_state.comment = ""
        st.session_state.comment_applied = False

        st.success("리포트 준비 완료. 좌측 ‘동작 분석 결과’ 탭에서 확인하세요.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── 2) 동작 분석 결과 탭 ───────────────────────────────────────────────────
elif nav == "동작 분석 결과":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>동작 분석 결과</div>", unsafe_allow_html=True)

    if not st.session_state.report_ready:
        st.info("먼저 ‘파일 업로드’ 탭에서 분석을 생성하세요.")
    else:
        st.markdown(f"**측면: {st.session_state.side_label}**  |  **파일:** {st.session_state.npz_name}")
        st.table(lines_to_df(st.session_state.rows))

        # 치료사 코멘트
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='h3'>치료사 코멘트</div>", unsafe_allow_html=True)
        st.session_state.comment = st.text_area(
            "", value=st.session_state.comment, height=120,
            placeholder="환자의 특징, 주의사항, 연습 방법, 다음 단계 권고 등을 입력"
        )
        if st.button("코멘트 저장", use_container_width=True):
            c = st.session_state.comment.strip()
            if c:
                st.session_state.report_text = (
                    f"■ {st.session_state.side_label} 다리\n" + "\n".join(st.session_state.rows)
                    + "\n\n■ 치료사 코멘트\n"
                    + "\n".join(f"⚠️ {line.strip()}" for line in c.splitlines() if line.strip())
                )
                st.session_state.comment_applied = True
                st.success("코멘트를 리포트에 추가했습니다. ‘최종 리포트’ 탭에서 확인하세요.")
            else:
                st.warning("코멘트가 비어 있습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

# ── 3) 최종 리포트 탭 ──────────────────────────────────────────────────────
elif nav == "최종 리포트":
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='h3'>최종 동작 분석 리포트</div>", unsafe_allow_html=True)

    if not st.session_state.report_ready or not st.session_state.report_text:
        st.info("먼저 ‘파일 업로드’에서 리포트를 생성하고 ‘동작 분석 결과’에서 코멘트를 저장하세요.")
    else:
        st.markdown("<div class='final'>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>측면: <b>{st.session_state.side_label}</b> · 파일: {st.session_state.npz_name}</div>", unsafe_allow_html=True)
        st.text(st.session_state.report_text)

        txt_buf = io.BytesIO(st.session_state.report_text.encode("utf-8"))
        st.download_button(
            "리포트 다운로드(.txt)", data=txt_buf.getvalue(),
            file_name=st.session_state.out_name, mime="text/plain",
            use_container_width=True
        )
        st.markdown("<div class='small'>※ 교육·임상 보조용 요약 리포트입니다.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
