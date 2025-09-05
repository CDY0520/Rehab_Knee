"""
파일명: src/qmetrics.py
 업로드된 영상에서 MediaPipe Pose로 샘플 프레임을 추출·추정하여 영상 품질(Q-metrics)을 계산한다.
 평균 가시성, 가시 비율, 바운딩 박스 비율, FPS, 가려짐 비율, 지터 표준편차를 산출하고 임계값으로 통과/재촬영을 판정한다.

블록 구성
 0) 라이브러리 임포트: numpy, OpenCV, mediapipe, json, pathlib
 1) 설정/임계값:
    - fps_min, avg_visibility, visible_ratio, bbox 범위, occlusion_max
 2) 핵심 함수 compute_qmetrics(video_path, sample_cap=150):
    - 비디오 오픈 → 균등 샘플 인덱스 생성
    - Pose 추정(실시간 설정) → 랜드마크 visibility/좌표 수집
    - 지표 계산: avg_visibility, visible_ratio, bbox_ratio_mean, occlusion_rate, jitter_std, fps
    - 통과 여부(pass) 판정
 3) 유틸: save_qmetrics_json(metrics, out_path)
 4) main/CLI(옵션): --video, --out, --sample-cap, --show 로 단독 실행 테스트

사용 방법
 1) 모듈 사용:
    from src.qmetrics import compute_qmetrics
    m = compute_qmetrics("results/raw/gait_side_NA_s001_2025-09-05.mp4")
 2) CLI 테스트:
    python src/qmetrics.py --video results/raw/sample.mp4 --out results/json/qmetrics_sample.json
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe가 필요합니다. `pip install mediapipe opencv-python numpy`") from e

# ------------------------------------------------------------
# 1) 설정/임계값 (문헌·관행 기반 권장값)
# ------------------------------------------------------------
THRESHOLDS = {
    "fps_min": 24.0,           # 권장 30, 최소 24
    "avg_visibility": 0.60,    # 평균 visibility
    "visible_ratio": 0.80,     # visibility>=0.5 비율
    "bbox_min": 0.40,          # 프레임 대비 인체 bbox 비율 하한
    "bbox_max": 0.90,          # 상한(너무 가까움)
    "occlusion_max": 0.10      # 포즈 미검출/심각 가림 프레임 비율
}


# ------------------------------------------------------------
# 내부 유틸
# ------------------------------------------------------------
def _uniform_indices(total_frames: int, cap: int) -> np.ndarray:
    if total_frames <= 0:
        return np.array([], dtype=int)
    n = min(cap, total_frames)
    return np.linspace(0, total_frames - 1, num=n, dtype=int)


def _bbox_ratio(xy: np.ndarray, frame_w: float, frame_h: float) -> float:
    xs = xy[:, 0] * frame_w
    ys = xy[:, 1] * frame_h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    bw = max(x1 - x0, 1.0)
    bh = max(y1 - y0, 1.0)
    return float((bw * bh) / max(frame_w * frame_h, 1.0))


# ------------------------------------------------------------
# 2) 핵심: Q-metrics 계산
# ------------------------------------------------------------
def compute_qmetrics(video_path: str, sample_cap: int = 150) -> Dict[str, Any]:
    """
    입력: 비디오 경로
    출력: Q-metrics dict
      - fps, frames_total, frames_sampled
      - avg_visibility, visible_ratio, bbox_ratio_mean
      - occlusion_rate, jitter_std
      - pass (임계값 만족 여부)
    """
    path = Path(video_path)
    if not path.exists():
        return {"error": "file_not_found", "path": str(path)}

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"error": "open_failed", "path": str(path)}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0.0)
    height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0.0)

    idxs = _uniform_indices(total_frames, sample_cap)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,          # 동영상 추정
        model_complexity=1,               # 속도 우선
        enable_segmentation=False,
        smooth_landmarks=True             # 지터 감소
    )

    vis_list: List[float] = []
    vis_ratio_list: List[float] = []
    bbox_ratio_list: List[float] = []
    deltas: List[float] = []
    occluded = 0
    prev_xy = None

    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if not res.pose_landmarks:
            occluded += 1
            continue

        lm = res.pose_landmarks.landmark
        vis = np.array([p.visibility for p in lm], dtype=np.float32)
        xy = np.array([[p.x, p.y] for p in lm], dtype=np.float32)

        vis_list.append(float(vis.mean()))
        vis_ratio_list.append(float((vis >= 0.5).mean()))
        bbox_ratio_list.append(_bbox_ratio(xy, width, height))

        if prev_xy is not None:
            deltas.append(float(np.linalg.norm((xy - prev_xy), axis=1).mean()))
        prev_xy = xy

        if vis.mean() < 0.2:
            occluded += 1

    cap.release()
    pose.close()

    sampled = max(len(vis_list), 0)
    occlusion_rate = float(occluded / max(len(idxs), 1))
    jitter_std = float(np.std(deltas)) if deltas else 0.0

    avg_visibility = float(np.mean(vis_list)) if vis_list else 0.0
    visible_ratio = float(np.mean(vis_ratio_list)) if vis_ratio_list else 0.0
    bbox_ratio_mean = float(np.mean(bbox_ratio_list)) if bbox_ratio_list else 0.0

    passed = (
        (fps >= THRESHOLDS["fps_min"]) and
        (sampled > 0) and
        (avg_visibility >= THRESHOLDS["avg_visibility"]) and
        (visible_ratio >= THRESHOLDS["visible_ratio"]) and
        (THRESHOLDS["bbox_min"] <= bbox_ratio_mean <= THRESHOLDS["bbox_max"]) and
        (occlusion_rate <= THRESHOLDS["occlusion_max"])
    )

    return {
        "path": str(path),
        "fps": round(fps, 2),
        "frames_total": int(total_frames),
        "frames_sampled": int(sampled),
        "avg_visibility": round(avg_visibility, 3),
        "visible_ratio": round(visible_ratio, 3),
        "bbox_ratio_mean": round(bbox_ratio_mean, 3),
        "occlusion_rate": round(occlusion_rate, 3),
        "jitter_std": round(jitter_std, 4),
        "pass": bool(passed),
        "thresholds": THRESHOLDS,
    }


# ------------------------------------------------------------
# 3) 저장 유틸
# ------------------------------------------------------------
def save_qmetrics_json(metrics: Dict[str, Any], out_path: str | Path) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return out


# ------------------------------------------------------------
# 4) main/CLI (옵션)
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse, time as _t

    parser = argparse.ArgumentParser(description="Compute Q-metrics from a video.")
    parser.add_argument("--video", required=True, help="input video path")
    parser.add_argument("--out", default=None, help="results/json/xxx.json")
    parser.add_argument("--sample-cap", type=int, default=150)
    args = parser.parse_args()

    m = compute_qmetrics(args.video, sample_cap=args.sample_cap)
    print(json.dumps(m, ensure_ascii=False, indent=2))

    if args.out:
        ts = int(_t.time())
        out = Path(args.out)
        if out.is_dir() or str(out).endswith("/"):
            out = out / f"qmetrics_{ts}.json"
        save_qmetrics_json(m, out)
        print(f"[saved] {out}")
