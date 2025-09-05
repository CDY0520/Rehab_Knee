"""
# 파일: src/app.py
# 목적: 무릎 중심 재활 피드백 시스템 데모 (Streamlit 앱)
#       - 한 페이지에서 5개 탭으로 구성
#         ① 영상 업로드 → ② 동작 분석 → ③ 홈 운동 처방
#         ④ 피드백 → ⑤ 모니터링
#
# 설계 포인트:
#  - Mediapipe Pose 기반 CSV/JSON 처리 연동
#  - metrics / events_* 모듈과 통합 (없을 경우 더미 분석기로 대체)
#  - 세션별 meta.json / analysis_summary.json / prescription.json / feedback.json 저장
#  - Streamlit UI에서 업로드, 분석, 처방, 피드백, 모니터링까지 일관된 파이프라인 제공
#
# 코드 블록 구성:
#   0) 라이브러리 임포트 및 경로 세팅
#   1) 외부 모듈 동적 임포트 (pose_to_csv, metrics, events 등)
#   2) 데이터클래스 정의 (SessionMeta, AnalysisResult)
#   3) 유틸 함수 (비디오 메타 읽기, JSON/CSV 저장·로드)
#   4) 더미 분석기 (모듈이 없을 경우 대비)
#   5) 규칙 기반 처방 / 피드백 생성 로직
#   6) Streamlit UI 구성 (사이드바 세션 설정 + 5개 탭)
#   7) 각 탭별 기능
#        - Tab1: 영상 업로드 & 메타 저장
#        - Tab2: 동작 분석 (pose 추출, 품질체크, 지표산출, 그래프)
#        - Tab3: 홈 운동 처방 (JSON 카드 생성/저장)
#        - Tab4: 피드백 (문장/레벨 메시지 생성)
#        - Tab5: 모니터링 (누적 세션 불러오기, 지표 추이 시각화)
#
# 사용 예시:
#   streamlit run src/app.py
#
# 입력:
#   - 업로드된 MP4 동영상
#   - 세션 메타데이터(task_type, side, fps, notes 등)
#
# 출력:
#   - results/<session_id>/ 폴더에 JSON/CSV 파일 저장
#   - Streamlit 화면에서 요약 지표, 처방 카드, 피드백 메시지, 모니터링 그래프 확인 가능
"""

from __future__ import annotations
import os
import io
import json
import time
import uuid
import shutil
import traceback
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple, List

import streamlit as st

# (선택) OpenCV로 동영상 메타 읽기용 - 설치 안 되어 있으면 자동 우회
try:
    import cv2
except Exception:
    cv2 = None

# (선택) 판다스/넘파이 - 결과 테이블/간단 계산
import pandas as pd
import numpy as np

# (선택) 시각화 - 간단 트렌드
import altair as alt

# =========================
# 0) 프로젝트 루트/경로 세팅
# -------------------------
# - PyCharm 구조를 기준: repo_root/ (여기에 src/, data/, results/ 가 있다고 가정)
# - Streamlit 실행 위치가 어디든 안정적으로 경로 잡히도록 처리
# =========================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR   = os.path.join(REPO_ROOT, "src")
DATA_DIR  = os.path.join(REPO_ROOT, "data")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# =========================
# 1) 외부 모듈 동적 임포트(있으면 사용/없으면 우회)
# -------------------------
# - 네가 만든 파일명 기준으로 시도
# - 실패해도 앱이 죽지 않도록 try/except
# =========================
def _try_imports():
    modules = {}
    try:
        from config import DEFAULT_FPS  # 선택: 네가 정의했다면 사용
        modules["DEFAULT_FPS"] = DEFAULT_FPS
    except Exception:
        modules["DEFAULT_FPS"] = 30

    try:
        # 예: pose 추출 함수 시그니처 가정
        # def extract_pose_to_csv(video_path:str, out_csv:str, side:str, task_type:str) -> Dict
        from pose_to_csv import extract_pose_to_csv
        modules["extract_pose_to_csv"] = extract_pose_to_csv
    except Exception:
        modules["extract_pose_to_csv"] = None

    # 이벤트 검출 (보행/STS/운동 등)
    for name in ["events", "events_gait", "events_sts", "events_exercise"]:
        try:
            modules[name] = __import__(name)
        except Exception:
            modules[name] = None

    try:
        # 예: compute_metrics(df)->Dict 형태 가정
        from metrics import compute_metrics
        modules["compute_metrics"] = compute_metrics
    except Exception:
        modules["compute_metrics"] = None

    # (선택) 품질 지표/미리보기 등
    try:
        from video_quality_pose import quick_quality_check  # 시그니처 가정
        modules["quick_quality_check"] = quick_quality_check
    except Exception:
        modules["quick_quality_check"] = None

    return modules

MODULES = _try_imports()

# =========================
# 2) 데이터클래스: 세션 메타/결과
# =========================
@dataclass
class SessionMeta:
    session_id: str
    created_at: float
    task_type: str               # 'gait' | 'sts' | 'exercise' | 'monitoring'
    side: str                    # 'LEFT' | 'RIGHT' | 'BOTH' | 'NA'
    facing: str                  # 'front' | 'back' | 'left' | 'right'
    fps_hint: int
    video_path: str              # 저장된 원본 경로
    notes: str = ""

@dataclass
class AnalysisResult:
    summary: Dict[str, Any]
    events: Dict[str, Any]
    metrics: Dict[str, Any]
    csv_path: Optional[str] = None
    json_path: Optional[str] = None

# =========================
# 3) 유틸: 동영상 메타 읽기
# -------------------------
# - OpenCV 있으면 정확히 읽고, 없으면 간단 우회
# =========================
def read_video_meta(video_path: str) -> Dict[str, Any]:
    meta = {"fps": None, "width": None, "height": None, "frame_count": None, "duration_sec": None}
    if cv2 is None:
        # 우회: 파일 크기만 제공
        try:
            size = os.path.getsize(video_path)
        except Exception:
            size = None
        meta.update({"fps": None, "width": None, "height": None, "frame_count": None, "duration_sec": None, "file_size": size})
        return meta

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return meta
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    meta["fps"] = round(float(fps), 2) if fps else None
    meta["width"] = w
    meta["height"] = h
    meta["frame_count"] = n
    meta["duration_sec"] = round(n / fps, 2) if fps and n else None
    return meta

# =========================
# 4) 유틸: 결과 저장/불러오기
# =========================
def session_dir(session_id: str) -> str:
    d = os.path.join(RESULTS_DIR, session_id)
    os.makedirs(d, exist_ok=True)
    return d

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_all_sessions() -> List[Tuple[str, Dict[str, Any]]]:
    """모니터링 탭용: 결과 폴더 스캔하여 세션 메타/요약 로드"""
    out = []
    if not os.path.isdir(RESULTS_DIR):
        return out
    for sid in sorted(os.listdir(RESULTS_DIR)):
        meta_p = os.path.join(RESULTS_DIR, sid, "meta.json")
        summary_p = os.path.join(RESULTS_DIR, sid, "analysis_summary.json")
        if os.path.isfile(meta_p):
            try:
                meta = json.load(open(meta_p, encoding="utf-8"))
                summ = json.load(open(summary_p, encoding="utf-8")) if os.path.isfile(summary_p) else {}
                out.append((sid, {"meta": meta, "summary": summ}))
            except Exception:
                continue
    return out

# =========================
# 5) 더미 분석기(백업 루틴)
# -------------------------
# - 실제 모듈이 없을 때도 앱 흐름 검증 가능
# - 무릎 중심 간단 지표 ROM/peak flex 등 랜덤 생성
# =========================
def dummy_analyze(csv_path: Optional[str], task_type: str, side: str) -> AnalysisResult:
    # 랜덤/규칙 기반의 간단 지표 생성 (데모용)
    rng = np.random.default_rng(seed=42)
    summary = {
        "HS": int(rng.integers(10, 30)) if task_type == "gait" else 0,
        "TO": int(rng.integers(10, 30)) if task_type == "gait" else 0,
        "ROM_knee": round(float(rng.uniform(25, 60)), 2),
        "peak_flex_swing": round(float(rng.uniform(35, 70)), 2) if task_type == "gait" else None,
        "Dorsi_max": round(float(rng.uniform(5, 25)), 2),
        "Plantar_max": round(float(rng.uniform(10, 40)), 2),
        "Cadence": round(float(rng.uniform(80, 120)), 1) if task_type == "gait" else None,
    }
    events = {"HS": summary["HS"], "TO": summary["TO"]}
    metrics = {"knee_hyperext_count": int(rng.integers(0, 2)),
               "valgus_varus_score": round(float(rng.uniform(-5, 5)), 2),
               "stance_swing_ratio": round(float(rng.uniform(0.9, 1.5)), 2) if task_type == "gait" else None}

    return AnalysisResult(summary=summary, events=events, metrics=metrics, csv_path=csv_path)

# =========================
# 6) 규칙 기반 처방/피드백
# -------------------------
# - 무릎 과신전/정렬/ROM 등 간단 규칙
# - 실제 임상 규칙은 metrics.py와 연동 가능
# =========================
def rule_based_prescription(analysis: AnalysisResult, meta: SessionMeta) -> Dict[str, Any]:
    m = analysis.metrics
    s = analysis.summary
    card: Dict[str, Any] = dict()

    # 기본 권장 세트/빈도
    base_sets = 3
    base_reps = 12

    # 규칙 1: 과신전(Back Knee) 많으면 햄스트링/둔근 강화 + 무릎 잠김 회피 cue
    if m.get("knee_hyperext_count", 0) >= 1:
        card["과신전 개선"] = {
            "운동": ["힙힌지(덤벨/밴드)", "브릿지 홀드", "Nordic Hamstring(쉬운 변형)"],
            "세트x반복": f"{base_sets} x 10~12",
            "주의": "무릎 잠그지 않기, 발뒤꿈치 체중 분배, 마지막 5도에서 잠김 방지 큐",
        }

    # 규칙 2: 정렬 편차(내반/외반) 절대값↑ → 중둔근/발목 안정화
    if abs(m.get("valgus_varus_score", 0)) > 3:
        card["정렬 개선(내반/외반)"] = {
            "운동": ["클램쉘", "사이드스텝 밴드 워크", "Single-Leg Balance with Reach"],
            "세트x반복": f"{base_sets} x {base_reps}",
            "주의": "무릎-발끝 정렬 유지, 거울 피드백 활용",
        }

    # 규칙 3: ROM이 낮으면(예: <35°) 슬관절 가동성/굴곡 강화
    if s.get("ROM_knee") and s["ROM_knee"] < 35:
        card["ROM 향상"] = {
            "운동": ["Heel Slide", "Wall Slide Squat(얕게)", "수동 슬관절 굴곡 스트레치"],
            "세트x반복": f"{base_sets} x {base_reps}",
            "주의": "통증 3/10 이하 범위, 얼음/온열 병행",
        }

    # 기본 권고 (항목 하나도 안 걸리면)
    if not card:
        card["유지 관리"] = {
            "운동": ["Sit-to-Stand 10회 x 3세트", "미니 스쿼트", "앵클 펌프"],
            "세트x반복": f"{base_sets} x {base_reps}",
            "주의": "천천히, 정렬 유지, 통증 증가 시 중단",
        }
    return card

def rule_based_feedback(analysis: AnalysisResult, meta: SessionMeta) -> Dict[str, Any]:
    m, s = analysis.metrics, analysis.summary
    msgs = []
    # 간단한 문장 템플릿
    if m.get("knee_hyperext_count", 0) >= 1:
        msgs.append("무릎을 끝까지 펴서 잠그지 마세요. 마지막 5도에서 멈추고 엉덩이 힘을 사용하세요.")
    if abs(m.get("valgus_varus_score", 0)) > 3:
        if m["valgus_varus_score"] > 0:
            msgs.append("무릎이 바깥으로 치우칩니다. 무릎-발끝이 같은 방향을 보도록 정렬하세요.")
        else:
            msgs.append("무릎이 안쪽으로 말립니다(Valgus). 중둔근에 힘을 주고 발 아치 유지하세요.")
    if s.get("ROM_knee") and s["ROM_knee"] < 35:
        msgs.append("무릎 굴곡 범위가 부족합니다. 통증 허용 범위에서 슬라이드/스트레치를 병행하세요.")

    # 심플 레벨 메시지
    if len(msgs) == 0:
        level = "🟢 정상 - 이상치 없음"
    elif len(msgs) == 1:
        level = "🟡 주의 - 이상치 1회 발생"
    else:
        level = "🔴 경고 - 이상치 2회 이상 발생"

    return {"level": level, "messages": msgs or ["오늘 동작은 안정적입니다. 현재 루틴을 유지하세요."]}

# =========================
# 7) 스트림릿 UI 시작
# =========================
st.set_page_config(page_title="Rehab Knee: 무릎 중심 재활 피드백", layout="wide")

st.title("재활 홈운동 관리: 무릎 집중 피드백")
st.caption("Mediapipe + Python 기반 / 5탭 한 페이지 구성")

# --- 사이드바: 세션 컨텍스트 설정 ---
with st.sidebar:
    st.header("세션 컨텍스트")
    # 최초 한 번 세션 ID 생성
    if "session_id" not in st.session_state:
        st.session_state.session_id = time.strftime("%Y%m%d") + "-" + uuid.uuid4().hex[:6]

    task_type = st.selectbox("과제 유형(task_type)", ["gait", "sts", "exercise", "monitoring"], index=0)
    side = st.selectbox("측면(side)", ["LEFT", "RIGHT", "BOTH", "NA"], index=0)
    facing = st.selectbox("촬영 방향", ["front", "back", "left", "right"], index=0)
    fps_hint = st.number_input("FPS 힌트", min_value=1, max_value=240, value=int(MODULES["DEFAULT_FPS"]), step=1)
    notes = st.text_input("메모(Optional)", "")

    st.markdown("---")
    st.markdown(f"**세션 ID:** `{st.session_state.session_id}`")
    st.markdown(f"결과 폴더: `{session_dir(st.session_state.session_id)}`")

# --- 5개 탭 레이아웃 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 영상 업로드", "② 동작 분석", "③ 홈 운동 처방", "④ 피드백", "⑤ 모니터링"])

# =========================
# 탭 1) 영상 업로드
# - mp4 업로드 → repo/results/<session_id>/original.mp4 저장
# - 메타데이터 추출 후 meta.json 저장
# - 이후 파이프라인에서 공통 사용
# =========================
with tab1:
    st.subheader("① 영상 업로드 (세션 메타 설정 & 원본 보관)")
    upl = st.file_uploader("MP4 동영상을 업로드하세요", type=["mp4"])

    colA, colB = st.columns([1,1])
    with colA:
        if upl is not None:
            st.video(upl)

    if upl is not None and st.button("이 영상으로 세션 시작/저장", use_container_width=True):
        # 세션 폴더 준비
        sdir = session_dir(st.session_state.session_id)
        video_path = os.path.join(sdir, "original.mp4")
        with open(video_path, "wb") as f:
            f.write(upl.read())

        # 메타 생성/저장
        meta = SessionMeta(
            session_id=st.session_state.session_id,
            created_at=time.time(),
            task_type=task_type, side=side, facing=facing,
            fps_hint=int(fps_hint), video_path=video_path, notes=notes
        )
        meta_d = asdict(meta)
        meta_d["video_meta"] = read_video_meta(video_path)
        save_json(meta_d, os.path.join(sdir, "meta.json"))

        st.success("세션 생성 및 원본 저장 완료!")
        st.json(meta_d)

    # 저장된 메타 미리보기
    meta_p = os.path.join(session_dir(st.session_state.session_id), "meta.json")
    if os.path.isfile(meta_p):
        st.info("현재 세션 메타")
        st.json(json.load(open(meta_p, encoding="utf-8")))

# =========================
# 탭 2) 동작 분석
# - Mediapipe Pose → CSV/JSON (있는 모듈 우선) + 품질체크/이벤트/지표
# - summary/metrics/events 저장 및 다운로드
# =========================
with tab2:
    st.subheader("② 동작 분석 (Mediapipe 결과 기반 지표 산출)")

    sdir = session_dir(st.session_state.session_id)
    meta_p = os.path.join(sdir, "meta.json")
    if not os.path.isfile(meta_p):
        st.warning("먼저 ① 영상 업로드 탭에서 세션을 생성하세요.")
    else:
        meta_d = json.load(open(meta_p, encoding="utf-8"))
        video_path = meta_d.get("video_path")

        # 2-1) Pose → CSV
        csv_path = os.path.join(sdir, "pose.csv")
        log_box = st.empty()
        try:
            if MODULES["extract_pose_to_csv"] is not None:
                log_box.info("Mediapipe Pose 추출 중...")
                _ = MODULES["extract_pose_to_csv"](
                    video_path=video_path,
                    out_csv=csv_path,
                    side=meta_d["side"],
                    task_type=meta_d["task_type"],
                )
            else:
                # 더미 CSV 생성(프레임/각도 예시 컬럼)
                log_box.warning("extract_pose_to_csv 모듈이 없어 더미 CSV를 생성합니다.")
                df_dummy = pd.DataFrame({
                    "frame": np.arange(0, 300),
                    "knee_angle": 30 + 15*np.sin(np.linspace(0, 6.28, 300)),
                    "ankle_angle": 10 + 10*np.sin(np.linspace(0, 9.42, 300)),
                    "visibility": np.clip(np.random.normal(0.8, 0.05, 300), 0, 1)
                })
                df_dummy.to_csv(csv_path, index=False)
            log_box.success("Pose CSV 생성 완료")
        except Exception as e:
            st.error("Pose 추출에 실패했습니다.")
            st.code(traceback.format_exc())

        # 2-2) 품질 체크(선택)
        qual = {}
        if os.path.isfile(csv_path):
            try:
                if MODULES["quick_quality_check"] is not None:
                    qual = MODULES["quick_quality_check"](video_path=video_path, csv_path=csv_path)
                else:
                    qual = {"avg_visibility": round(float(pd.read_csv(csv_path)["visibility"].mean()), 3)}
            except Exception:
                qual = {}

        # 2-3) 이벤트/지표 계산
        try:
            # 네가 만든 metrics.py가 있으면 사용
            if MODULES["compute_metrics"] is not None:
                df = pd.read_csv(csv_path)
                metrics = MODULES["compute_metrics"](df)
                # summary/events는 개별 events_* 모듈에서 가져온다고 가정
                summary, events = {}, {}
            else:
                # 백업 더미
                analysis = dummy_analyze(csv_path, meta_d["task_type"], meta_d["side"])
                metrics, summary, events = analysis.metrics, analysis.summary, analysis.events
        except Exception:
            analysis = dummy_analyze(csv_path, meta_d["task_type"], meta_d["side"])
            metrics, summary, events = analysis.metrics, analysis.summary, analysis.events

        # 2-4) 저장 및 다운로드 리소스 생성
        analysis_summary = {"summary": summary, "events": events, "metrics": metrics, "quality": qual}
        summary_p = os.path.join(sdir, "analysis_summary.json")
        save_json(analysis_summary, summary_p)

        st.success("분석 완료!")
        st.write("**요약/지표 미리보기**")
        st.json(analysis_summary)

        # 2-5) 간단 그래프(예: 무릎 각도 시계열)
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            if "knee_angle" in df.columns:
                st.write("무릎 각도 트렌드(예시)")
                chart = alt.Chart(df.reset_index()).mark_line().encode(
                    x=alt.X('index:Q', title='frame'),
                    y=alt.Y('knee_angle:Q', title='deg')
                ).properties(height=200, use_container_width=True)
                st.altair_chart(chart, use_container_width=True)

        # 2-6) 다운로드 버튼
        st.download_button("요약 JSON 다운로드", data=json.dumps(analysis_summary, ensure_ascii=False, indent=2),
                           file_name="analysis_summary.json", mime="application/json")
        if os.path.isfile(csv_path):
            st.download_button("Pose CSV 다운로드", data=open(csv_path, "rb").read(),
                               file_name="pose.csv", mime="text/csv")

# =========================
# 탭 3) 홈 운동 처방
# - 분석 요약을 바탕으로 rule_based 처방 카드 생성
# - JSON/CSV로 저장/다운로드
# =========================
with tab3:
    st.subheader("③ 홈 운동 처방 (분석 기반 카드 생성)")

    sdir = session_dir(st.session_state.session_id)
    summary_p = os.path.join(sdir, "analysis_summary.json")
    if not os.path.isfile(summary_p):
        st.warning("② 동작 분석을 먼저 수행하세요.")
    else:
        meta_d = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
        summ_d = json.load(open(summary_p, encoding="utf-8"))

        analysis = AnalysisResult(summary=summ_d.get("summary", {}), events=summ_d.get("events", {}),
                                  metrics=summ_d.get("metrics", {}))
        meta = SessionMeta(**{k: meta_d[k] for k in ["session_id","created_at","task_type","side","facing","fps_hint","video_path","notes"]})
        card = rule_based_prescription(analysis, meta)

        st.success("처방 카드 생성")
        for k, v in card.items():
            with st.expander(f"· {k}", expanded=True):
                st.write("**운동:** ", ", ".join(v.get("운동", [])))
                st.write("**세트x반복:** ", v.get("세트x반복"))
                st.write("**주의:** ", v.get("주의"))

        # 저장 + 다운로드
        card_p = os.path.join(sdir, "prescription.json")
        save_json(card, card_p)
        st.download_button("처방 카드(JSON) 다운로드",
                           data=json.dumps(card, ensure_ascii=False, indent=2),
                           file_name="prescription.json", mime="application/json")

# =========================
# 탭 4) 피드백
# - 간단 문장형 피드백 자동 생성 (레벨/문장)
# - LLM 연결은 옵션(지금은 규칙 기반)
# =========================
with tab4:
    st.subheader("④ 피드백 (임상 포인트/경고 메시지 자동 생성)")

    sdir = session_dir(st.session_state.session_id)
    summary_p = os.path.join(sdir, "analysis_summary.json")
    if not os.path.isfile(summary_p):
        st.warning("② 동작 분석을 먼저 수행하세요.")
    else:
        meta_d = json.load(open(os.path.join(sdir, "meta.json"), encoding="utf-8"))
        summ_d = json.load(open(summary_p, encoding="utf-8"))

        analysis = AnalysisResult(summary=summ_d.get("summary", {}), events=summ_d.get("events", {}),
                                  metrics=summ_d.get("metrics", {}))
        meta = SessionMeta(**{k: meta_d[k] for k in ["session_id","created_at","task_type","side","facing","fps_hint","video_path","notes"]})
        fb = rule_based_feedback(analysis, meta)

        st.write(f"**통합 레벨:** {fb['level']}")
        for msg in fb["messages"]:
            st.markdown(f"- {msg}")

        # 저장 + 다운로드
        fb_p = os.path.join(sdir, "feedback.json")
        save_json(fb, fb_p)
        st.download_button("피드백(JSON) 다운로드",
                           data=json.dumps(fb, ensure_ascii=False, indent=2),
                           file_name="feedback.json", mime="application/json")

# =========================
# 탭 5) 모니터링
# - 누적 세션 요약 로드 → 트렌드/리포트
# - CSV/JSON 업로드도 지원(향후)
# =========================
with tab5:
    st.subheader("⑤ 모니터링 (세션 추이/성과 확인)")

    sessions = load_all_sessions()
    if not sessions:
        st.info("아직 저장된 세션이 없습니다. ①~④ 단계를 먼저 수행하세요.")
    else:
        # 리스트/선택
        sid_list = [sid for sid, _ in sessions]
        sel = st.selectbox("세션 선택", options=sid_list, index=len(sid_list)-1)
        sel_item = next(v for s, v in sessions if s == sel)

        st.markdown("**선택 세션 메타**")
        st.json(sel_item["meta"])

        st.markdown("**선택 세션 요약**")
        st.json(sel_item["summary"])

        # 간단 추이: 최근 N개 세션의 ROM_knee 트렌드
        rows = []
        for sid, item in sessions:
            summ = item.get("summary", {})
            rom = (summ.get("summary") or {}).get("ROM_knee")
            if rom is not None:
                rows.append({"session": sid, "ROM_knee": rom, "created_at": item["meta"].get("created_at", 0)})
        if rows:
            df_hist = pd.DataFrame(rows).sort_values("created_at")
            st.write("ROM_knee 변화(세션별)")
            chart = alt.Chart(df_hist).mark_line(point=True).encode(
                x=alt.X('session:N', sort=None),
                y=alt.Y('ROM_knee:Q')
            ).properties(height=220, use_container_width=True)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("ROM 지표가 누적된 세션이 아직 충분치 않습니다.")
