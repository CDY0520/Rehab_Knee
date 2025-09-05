"""
파일명: app/pages/02_동작분석.py
 업로드된 keypoints(.npz)를 불러와 보행(HS/MS/TO)·STS(Seat-off/Full-stand) 이벤트를 검출하고,
 기본 지표(steps/cadence, cycles 등)를 표·그래프로 보여준다.
 JSON/CSV로 결과를 저장하고 다운로드를 지원한다.

블록 구성
 0) 임포트/경로: Streamlit, numpy, matplotlib, 프로젝트 루트 경로 추가
 1) 유틸:
    - 최근 .npz 파일 탐색, 요약 메타 표시
    - 시각화 헬퍼(라인 + 이벤트 수직선)
    - 저장/다운로드 헬퍼(JSON/CSV)
 2) UI 섹션:
    - 입력 선택: 보행/STS용 .npz, 보행 측 힌트(left/right), 파라미터(스무딩 창 등)
    - 보행 분석: detect_gait_events 호출 → 표·그래프 → 저장/다운로드
    - STS  분석: detect_sts_events  호출 → 표·그래프 → 저장/다운로드
 3) 유효성 체크: 보행 HS<6 또는 STS cycles<2 경고 표시
 4) 에러 처리: 파일 없음/불량 시 사용자 메시지

사용 방법
 1) 01_영상업로드에서 품질 통과 → 포즈추출(npz) 생성 후 본 탭에서 선택 분석
 2) 저장 위치: results/json/, 그래프는 results/figures/
"""

from __future__ import annotations
import sys
import io
import json
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 0) 경로/임포트
# -------------------------------
ROOT = Path(__file__).resolve().parents[2]  # 프로젝트 루트
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.events import (  # noqa: E402
    detect_gait_events,
    detect_sts_events,
    save_events_json,
    save_events_csv_timeline,
    L_ANKLE, R_ANKLE, L_HIP, R_HIP,
)

KEY_DIR = ROOT / "results" / "keypoints"
JSON_DIR = ROOT / "results" / "json"
FIG_DIR = ROOT / "results" / "figures"
for d in (KEY_DIR, JSON_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="동작분석", layout="wide")


# -------------------------------
# 1) 유틸
# -------------------------------
def list_npz(pattern: str | None = None) -> list[Path]:
    files = sorted(KEY_DIR.glob("*.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if pattern:
        files = [p for p in files if pattern in p.name]
    return files


def load_npz_meta(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    n = int(d["frames"].shape[0]) if "frames" in d else int(d["lm_x"].shape[0])
    fps = float(meta.get("fps", 0))
    dur = (n / fps) if fps > 0 else 0
    return {"frames": n, "fps": fps, "duration_sec": dur, "width": meta.get("width"), "height": meta.get("height")}


def plot_signal_with_events(
    t_ms: np.ndarray,
    y: np.ndarray,
    title: str,
    event_dict: dict[str, list[int]] | None = None,
) -> bytes:
    """matplotlib로 선+수직 이벤트 마커 렌더 → PNG bytes 반환"""
    fig, ax = plt.subplots(figsize=(8, 3))
    t = (t_ms - t_ms[0]) / 1000.0
    ax.plot(t, y, linewidth=1.2)
    if event_dict:
        colors = {"HS_ms": "tab:red", "TO_ms": "tab:orange", "MS_ms": "tab:green",
                  "seat_off_ms": "tab:purple", "full_stand_ms": "tab:blue"}
        for k, arr in event_dict.items():
            for tm in arr:
                ax.axvline((tm - t_ms[0]) / 1000.0, color=colors.get(k, "gray"), alpha=0.6, linestyle="--", linewidth=1)
    ax.set_xlabel("time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def npz_to_basic_signal(npz_path: Path, mode: str, side_hint: str = "left") -> tuple[np.ndarray, np.ndarray]:
    """가벼운 시각화를 위해: 보행=발목 y, STS=골반 y"""
    d = np.load(npz_path, allow_pickle=True)
    ly = d["lm_y"]
    t_ms = d["t_ms"]
    if mode == "gait":
        idx = L_ANKLE if side_hint.startswith("l") else R_ANKLE
        sig = ly[:, idx]
    else:
        sig = (ly[:, L_HIP] + ly[:, R_HIP]) / 2.0
    return t_ms, sig


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# -------------------------------
# 2) UI
# -------------------------------
st.title("동작분석")
st.caption("npz 포즈 시계열을 사용해 HS/MS/TO, Seat-off/Full-stand, 를 검출합니다.")

col_l, col_r = st.columns([1, 1])

# ---- 입력 섹션: 보행
with col_l:
    st.subheader("보행 분석")
    gait_files = list_npz("gait") or list_npz()  # 패턴 없으면 전체
    sel_gait = st.selectbox(
        "보행 .npz 파일 선택",
        options=gait_files,
        format_func=lambda p: p.name if isinstance(p, Path) else str(p),
        index=0 if gait_files else None,
    )
    side_hint = st.radio("측 힌트(side)", options=["left", "right"], horizontal=True, index=0)
    run_gait = st.button("보행 이벤트 검출", use_container_width=True, type="primary", disabled=not gait_files)

# ---- 입력 섹션: STS
with col_r:
    st.subheader("STS 분석")
    sts_files = list_npz("sts") or list_npz()
    sel_sts = st.selectbox(
        "STS .npz 파일 선택",
        options=sts_files,
        format_func=lambda p: p.name if isinstance(p, Path) else str(p),
        index=0 if sts_files else None,
    )
    run_sts = st.button("STS 이벤트 검출", use_container_width=True, type="primary", disabled=not sts_files)

st.divider()

# -------------------------------
# 3) 보행 분석
# -------------------------------
if run_gait and sel_gait:
    try:
        meta = load_npz_meta(sel_gait)
        st.info(f"파일: {sel_gait.name} | FPS {meta['fps']:.2f} | 길이 {meta['duration_sec']:.2f}s | 프레임 {meta['frames']}")

        result = detect_gait_events(str(sel_gait), side_hint=side_hint)
        ev = result["events"]
        mt = result["metrics"]

        # 요약 지표
        st.markdown("**보행 지표**")
        st.dataframe(
            {
                "steps": [mt["steps"]],
                "cadence(spm)": [mt["cadence_spm"]],
                "stance_ratio(mean)": [mt["stance_ratio_mean"]],
                "swing_ratio(mean)": [mt["swing_ratio_mean"]],
            },
            use_container_width=True,
        )

        # 타임라인 표
        st.markdown("**이벤트 타임라인(ms)**")
        st.dataframe(
            {
                "HS_ms": [ev["HS_ms"]],
                "TO_ms": [ev["TO_ms"]],
                "MS_ms": [ev["MS_ms"]],
            },
            use_container_width=True,
        )

        # 간단 시각화
        t_ms, sig = npz_to_basic_signal(sel_gait, mode="gait", side_hint=side_hint)
        png = plot_signal_with_events(t_ms, sig, f"Ankle-y with HS/TO/MS ({side_hint})", ev)
        st.image(png, caption="보행 신호 시각화", use_column_width=True)

        # 유효성 체크
        if mt["steps"] < 6:
            st.warning("검출된 HS 수가 적습니다(steps<6). 더 긴 보행 영상 권장.")

        # 저장/다운로드
        base = f"events_gait_{stamp()}"
        out_json = JSON_DIR / f"{base}.json"
        save_events_json(result, out_json)
        with out_json.open("rb") as f:
            st.download_button("보행 이벤트 JSON 다운로드", f, file_name=out_json.name, mime="application/json")

        out_csv = JSON_DIR / f"{base}.csv"
        save_events_csv_timeline(result, out_csv)
        with out_csv.open("rb") as f:
            st.download_button("보행 타임라인 CSV 다운로드", f, file_name=out_csv.name, mime="text/csv")

    except Exception as e:
        st.error(f"보행 분석 실패: {e}")

st.divider()

# -------------------------------
# 4) STS 분석
# -------------------------------
if run_sts and sel_sts:
    try:
        meta = load_npz_meta(sel_sts)
        st.info(f"파일: {sel_sts.name} | FPS {meta['fps']:.2f} | 길이 {meta['duration_sec']:.2f}s | 프레임 {meta['frames']}")

        result = detect_sts_events(str(sel_sts))
        ev = result["events"]
        mt = result["metrics"]

        # 요약 지표
        st.markdown("**STS 지표**")
        st.dataframe(
            {
                "cycles": [mt["cycles"]],
                "mean_cycle_sec": [mt["mean_cycle_sec"]],
            },
            use_container_width=True,
        )

        # 타임라인 표
        st.markdown("**이벤트 타임라인(ms)**")
        st.dataframe(
            {
                "seat_off_ms": [ev["seat_off_ms"]],
                "full_stand_ms": [ev["full_stand_ms"]],
            },
            use_container_width=True,
        )

        # 간단 시각화
        t_ms, sig = npz_to_basic_signal(sel_sts, mode="sts")
        png = plot_signal_with_events(t_ms, sig, "Pelvis-y with Seat-off/Full-stand", ev)
        st.image(png, caption="STS 신호 시각화", use_column_width=True)

        # 유효성 체크
        if mt["cycles"] < 2:
            st.warning("검출된 반복 횟수가 적습니다(cycles<2). STS 3회 이상 촬영 권장.")

        # 저장/다운로드
        base = f"events_sts_{stamp()}"
        out_json = JSON_DIR / f"{base}.json"
        save_events_json(result, out_json)
        with out_json.open("rb") as f:
            st.download_button("STS 이벤트 JSON 다운로드", f, file_name=out_json.name, mime="application/json")

        out_csv = JSON_DIR / f"{base}.csv"
        save_events_csv_timeline(result, out_csv)
        with out_csv.open("rb") as f:
            st.download_button("STS 타임라인 CSV 다운로드", f, file_name=out_csv.name, mime="text/csv")

    except Exception as e:
        st.error(f"STS 분석 실패: {e}")

# -------------------------------
# 5) 네비게이션
# -------------------------------
st.divider()
col_a, col_b = st.columns(2)
with col_a:
    st.link_button("업로드로 돌아가기", "01_영상업로드")
with col_b:
    st.link_button("다음 탭으로 이동(처방/모니터링)", "#")  # 추후 라우팅 연결
