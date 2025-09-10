"""
파일명: app/pages/01_영상업로드.py
 재활 프로젝트 스트림릿 ① 영상 업로드 페이지.
 4개 영상(보행 정면·측면, STS 정면·측면) 업로드 → 기본 메타(FPS/해상도/길이) 확인 →
 MediaPipe 기반 Q-metrics 계산(avg_visibility, visible_ratio, fps, occlusion_rate, jitter_std)
 → 통과/재촬영 판정 및 JSON 저장/다운로드까지 수행한다.

블록 구성
 0) 임포트/경로: pathlib/time/tempfile/OpenCV/Streamlit/pandas + qmetrics 모듈
 1) 유틸: 파일 저장(save_uploaded_file), 메타 추출(probe_video_meta), 파일명 규칙(build_filename)
 2) 입력 폼: 세션ID/날짜/거리/높이/기기 + 4개 업로드 슬롯
 3) 제출 처리:
    - 저장·메타 분석 → Q-metrics 계산 → 결과 표 + 판정(통과/재촬영)
    - results/json/ 에 요약 저장 및 다운로드 버튼
    - 측면 2개 모두 통과 시 “동작분석으로 이동” 버튼 활성
 4) 촬영 가이드: 문헌 근거 기반 권장 조건 고정 노출

사용 방법
 1) 페이지에서 4개 파일 업로드 → [저장·분석]
 2) 표에서 지표/판정 확인, JSON 다운로드
 3) 측면 두 클립 통과 시 다음 단계로 이동
"""

import time
from pathlib import Path
import tempfile
import cv2
import streamlit as st
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.qmetrics import compute_qmetrics, save_qmetrics_json

# -------------------------------
# 0) 경로·페이지 설정
# -------------------------------
RAW_DIR = Path("results") / "raw"       # 업로드 저장
JSON_DIR = Path("results") / "json"     # 리포트 저장
# mkdir는 저장 시 유틸에서 보장

st.set_page_config(page_title="영상 업로드", layout="wide")
st.title("영상 업로드")

st.caption(
    "측면 촬영 필수 · 영상 길이 최소 20초 · 동작 반복 최소 3회 · 전신 프레임 유지 · 삼각대 고정 · 거리 3 m / 높이 1.2 m 권장 "
)

# -------------------------------
# 1) 유틸 함수
# -------------------------------
def save_uploaded_file(uploaded_file: "UploadedFile", target_path: Path) -> Path:
    """스트림릿 UploadedFile을 로컬 파일로 저장"""
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(target_path)
    return target_path

def probe_video_meta(path: Path) -> dict:
    """OpenCV로 기본 메타(FPS, 프레임수, 해상도, 길이초) 추출"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"ok": False, "error": "open_failed"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frames / fps if fps > 0 else 0.0
    cap.release()
    return {"ok": True, "fps": round(float(fps), 2), "frames": frames, "width": w, "height": h, "duration_sec": round(duration, 2)}

def build_filename(task: str, view: str, side: str, session_id: str, date_str: str, ext: str) -> str:
    """파일명 규칙: task_view_side_session_date.ext  예) gait_side_NA_s001_2025-09-05.mp4"""
    safe = lambda s: "".join(c for c in s if c.isalnum() or c in ("-","_")).strip("_")
    return f"{safe(task)}_{safe(view)}_{safe(side)}_{safe(session_id)}_{safe(date_str)}{ext}"

# -------------------------------
# 2) 입력 폼
# -------------------------------
with st.form("meta_and_upload"):
    colm = st.columns(5)
    with colm[0]:
        session_id = st.text_input("세션 ID", value="s001")
    with colm[1]:
        shot_date = st.text_input("촬영 날짜", value=time.strftime("%Y-%m-%d"))
    with colm[2]:
        cam_dist = st.number_input("카메라 거리(m)", min_value=0.5, max_value=10.0, value=3.0, step=0.1)
    with colm[3]:
        cam_height = st.number_input("카메라 높이(m)", min_value=0.3, max_value=2.0, value=1.2, step=0.1)
    with colm[4]:
        device = st.text_input("촬영 기기", value="smartphone/tablet")

    st.markdown("#### 파일 업로드")
    c1, c2 = st.columns(2)
    with c1:
        gait_side = st.file_uploader("보행 · 측면(필수)", type=["mp4", "mov", "avi", "mkv"], key="gait_s")
        sts_side = st.file_uploader("STS · 측면(필수)", type=["mp4", "mov", "avi", "mkv"], key="sts_s")
    with c2:
        gait_front = st.file_uploader("보행 · 정면", type=["mp4", "mov", "avi", "mkv"], key="gait_f")
        sts_front = st.file_uploader("STS · 정면", type=["mp4", "mov", "avi", "mkv"], key="sts_f")


    submitted = st.form_submit_button("영상 품질 분석")

# -------------------------------
# 3) 제출 처리: 저장 → 메타 → Q-metrics → 표/저장
# -------------------------------
if submitted:
    if (gait_side is None) or (sts_side is None):
        st.error("측면 영상(보행·STS)은 필수입니다.")
        st.stop()

    uploads = {
        "gait_front": gait_front,
        "gait_side": gait_side,
        "sts_front": sts_front,
        "sts_side": sts_side,
    }

    rows = []
    q_results = {}
    for key, uf in uploads.items():
        if uf is None:
            rows.append({"클립": key, "상태": "미업로드"})
            continue

        task = "gait" if "gait" in key else "sts"
        view = "front" if "front" in key else "side"
        side = "NA"

        ext = Path(uf.name).suffix.lower() or ".mp4"
        fname = build_filename(task, view, side, session_id, shot_date, ext)
        target = RAW_DIR / fname

        saved = save_uploaded_file(uf, target)
        meta = probe_video_meta(saved)

        # Q-metrics 계산
        with st.spinner(f"{key} · Q-metrics 계산 중..."):
            qm = compute_qmetrics(str(saved))
        q_results[key] = qm

        rec = {"클립": key, "파일": str(saved)}
        if meta.get("ok") and "error" not in qm:
            rec.update({
                "상태": "업로드",
                "해상도": f'{meta["width"]}x{meta["height"]}',
                "FPS": qm.get("fps", meta["fps"]),
                "길이(초)": meta["duration_sec"],
                "avg_vis": qm.get("avg_visibility", 0),
                "vis_ratio": qm.get("visible_ratio", 0),
                "occlusion": qm.get("occlusion_rate", 0),
                "jitter_std": qm.get("jitter_std", 0),
                "판정": "통과 ✅" if qm.get("pass") else "재촬영 ⚠️",
            })
        else:
            rec.update({
                "상태": "분석 실패",
                "해상도": "-", "FPS": "-", "길이(초)": "-",
                "avg_vis": "-", "vis_ratio": "-",
                "occlusion": "-", "jitter_std": "-", "판정": "분석 실패"
            })
        rows.append(rec)

    st.success("업로드 · Q-metrics 분석 완료")

    st.markdown("### 영상 품질 분석 결과")
    # --- 결과 표(요청 컬럼만) ---
    ordered_cols = ["클립","상태","파일","해상도","FPS","길이(초)","avg_vis","vis_ratio",
                    "occlusion","jitter_std","판정"]
    df = pd.DataFrame(rows)[ordered_cols]
    st.dataframe(df, use_container_width=True)

    # --- 임계치/가이드 생성 로직 ---
    th = next((q.get("thresholds") for q in q_results.values()
               if isinstance(q, dict) and "thresholds" in q), None) or {}


    def fail_reasons(qm: dict) -> list[str]:
        if not qm or "error" in qm:
            return ["분석실패"]
        fails = []
        if qm.get("fps", 0) < th.get("fps_min", 24):                    fails.append(f'FPS<{th.get("fps_min", 24)}')
        if qm.get("avg_visibility", 1) < th.get("avg_visibility", .6):  fails.append("avg_vis 낮음")
        if qm.get("visible_ratio", 1) < th.get("visible_ratio", .8):    fails.append("vis_ratio 낮음")
        if qm.get("occlusion_rate", 0) > th.get("occlusion_max", .10):   fails.append("occlusion 높음")
        if qm.get("duration_sec", 0) < th.get("duration_min", 20):
            fails.append(f"영상 길이<{th.get('duration_min', 20)}s")
        return fails


    def guidance(miss: list[str]) -> list[str]:
        tips = []
        if any("avg_vis 낮음" in m for m in miss):
            tips.append("가시성 낮음 → 조명 밝게, 대비 색상, 가림 제거.")
        if any("vis_ratio 낮음" in m for m in miss):
            tips.append("가시성 낮음 → 조명 밝게, 대비 색상, 가림 제거.")
        if any("FPS<" in m for m in miss):
            tips.append("프레임레이트 낮음 → 30fps 설정, 슬로모션/저속 해제.")
        if any("occlusion 높음" in m for m in miss):
            tips.append("가림 많음 → 배경 정리, 삼각대 고정, 중앙 위치.")
        if any("영상 길이<" in m for m in miss):
            tips.append("영상 길이 부족 → 최소 20초 이상 촬영.")
        return tips or ["전신이 화면에 안정적으로 나오도록 재촬영."]

    # --- 재촬영 가이드: 측면(gait_side, sts_side) + 재촬영만 표로 출력 ---
    guide_rows = []
    for key in ("gait_side", "sts_side"):
        qm = q_results.get(key, {})
        if not qm or qm.get("pass", False):
            continue
        miss = fail_reasons(qm)
        guide_rows.append({
            "클립": key,
            "재촬영 사유": ", ".join(miss),
            "가이드": "\n".join(guidance(miss))
        })

    st.markdown("### 재촬영 가이드")
    if guide_rows:
        st.table(pd.DataFrame(guide_rows))  # 줄바꿈 반영됨, 옵션 없음
    else:
        st.write("측면 클립은 재촬영 권고 항목이 없습니다.")

    # --- JSON 저장 + 다운로드 ---
    ts = int(time.time())
    out_json = JSON_DIR / f"qmetrics_{session_id}_{ts}.json"
    save_qmetrics_json({"session": session_id, "date": shot_date, "results": q_results}, out_json)
    with out_json.open("rb") as f:
        st.download_button("품질 리포트 JSON 다운로드", f, file_name=out_json.name, mime="application/json")

    # --- 다음 단계 활성 조건 ---
    sides_ok = all(q_results.get(k, {}).get("pass") for k in ("gait_side", "sts_side") if q_results.get(k))
    if sides_ok:
        st.success("측면 두 영상 모두 통과. 다음 단계로 이동 가능합니다.")
        st.link_button("동작분석으로 이동", "02_동작분석")
    else:
        st.warning("측면 영상 중 ‘재촬영’ 판정이 있으면 동작분석 단계에서 제외됩니다.")


# -------------------------------
# 4) 촬영 가이드(근거 기반)
# -------------------------------
with st.expander("촬영 가이드"):
    st.write(
        "- 측면 촬영은 필수이며, 정면 촬영은 보조로 사용됩니다.\n"
        "- 삼각대를 고정하며, 전신이 항상 프레임 안에 있어야 합니다.\n"
        "- 영상 길이는 최소 20초 이상으로 촬영합니다.\n"
        "- 보행은 최소 10걸음, 동작은 최소 3회 반복하여 촬영합니다.\n"
        "- 촬영하는 대상자와의 거리는 3 m, 높이는 1.2 m 권장합니다.\n"
        "- 촬영 시 조명과 가림을 최소화 합니다."
    )
