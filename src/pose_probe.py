"""
파일명: src/pose_probe.py
 업로드된 영상을 Mediapipe Pose로 전체 프레임 추정하여
 랜드마크 시계열을 저장한다(.npz 기본, 옵션으로 .csv 병행).
 회전(메타데이터/수동), 좌우반전, 프레임 스킵 간격, 리사이즈를 지원한다.

블록 구성
 0) 임포트/상수: OpenCV, Mediapipe, numpy, argparse, pathlib, json
 1) 유틸:
    - exif 기반 자동 회전값 추정(가능 시), 수동 회전/좌우반전 적용
    - 타임스탬프(ms) 계산, 출력 경로 생성
 2) 핵심: probe_pose(video, every, ...)
    - 프레임 루프: 스킵 간격(every)로 추정 → (33, x/y/vis) 수집
    - per-frame valid 플래그와 bbox_ratio, fps 기록
 3) 저장: np.savez_compressed(npz), 옵션 csv 저장
 4) main/CLI: --video, --out, --csv, --every, --resize, --complexity,
               --autorotate, --rotate, --flip

사용 방법
 1) 기본:
    python src/pose_probe.py --video results/raw/gait_side_...mp4
 2) 프레임 스킵 2배(빠르게):
    python src/pose_probe.py --video data/samples/sample_walk.mp4 --every 2
 3) CSV도 저장:
    python src/pose_probe.py --video data/samples/sample_sts.mp4 --csv
"""

from __future__ import annotations
import json
from pathlib import Path
import argparse
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe가 필요합니다. pip install mediapipe opencv-python numpy") from e

KEY_DIR = Path("results") / "keypoints"
KEY_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------
# 1) 유틸
# ---------------------------
def _apply_rotate_flip(frame, rotate_deg: int = 0, flip: bool = False):
    if rotate_deg % 360 != 0:
        if rotate_deg % 360 == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotate_deg % 360 == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotate_deg % 360 == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame

def _bbox_ratio(xy: np.ndarray, w: int, h: int) -> float:
    xs = xy[:, 0] * w
    ys = xy[:, 1] * h
    bw = max(xs.max() - xs.min(), 1.0)
    bh = max(ys.max() - ys.min(), 1.0)
    return float((bw * bh) / max(w * h, 1.0))

# ---------------------------
# 2) 핵심
# ---------------------------
def probe_pose(
    video_path: str,
    out_npz: str | Path | None = None,
    save_csv: bool = False,
    every: int = 1,
    resize_width: int | None = None,
    model_complexity: int = 1,
    autorotate: bool = False,   # 자리만; OpenCV에서 EXIF 접근 제한적
    rotate: int = 0,
    flip: bool = False,
) -> Path:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(str(path))

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    frames = []
    t_ms = []
    vis_mean = []
    bbox_ratios = []
    valid = []

    lm_x = []
    lm_y = []
    lm_v = []

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % max(1, every) != 0:
            i += 1
            continue

        frame = _apply_rotate_flip(frame, rotate_deg=rotate, flip=flip)

        if resize_width and resize_width < frame.shape[1]:
            scale = resize_width / frame.shape[1]
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            x = np.array([p.x for p in lm], dtype=np.float32)
            y = np.array([p.y for p in lm], dtype=np.float32)
            v = np.array([p.visibility for p in lm], dtype=np.float32)
            lm_x.append(x)
            lm_y.append(y)
            lm_v.append(v)
            vis_mean.append(float(v.mean()))
            xy = np.stack([x, y], axis=1)
            bbox_ratios.append(_bbox_ratio(xy, frame.shape[1], frame.shape[0]))
            valid.append(True)
        else:
            lm_x.append(np.full(33, np.nan, dtype=np.float32))
            lm_y.append(np.full(33, np.nan, dtype=np.float32))
            lm_v.append(np.zeros(33, dtype=np.float32))
            vis_mean.append(0.0)
            bbox_ratios.append(0.0)
            valid.append(False)

        frames.append(i)
        t_ms.append(int((i / fps) * 1000) if fps > 0 else i)
        i += 1

    cap.release()
    pose.close()

    lm_x = np.stack(lm_x, axis=0) if lm_x else np.zeros((0, 33), np.float32)
    lm_y = np.stack(lm_y, axis=0) if lm_y else np.zeros((0, 33), np.float32)
    lm_v = np.stack(lm_v, axis=0) if lm_v else np.zeros((0, 33), np.float32)

    meta = {
        "video": str(path),
        "fps": fps,
        "frames_total": total,
        "frames_kept": len(frames),
        "width": w,
        "height": h,
        "every": every,
        "rotate": rotate,
        "flip": flip,
    }

    out_npz = Path(out_npz) if out_npz else (KEY_DIR / (path.stem + ".npz"))
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        lm_x=lm_x, lm_y=lm_y, lm_v=lm_v,
        frames=np.array(frames, dtype=np.int32),
        t_ms=np.array(t_ms, dtype=np.int32),
        vis_mean=np.array(vis_mean, dtype=np.float32),
        bbox_ratio=np.array(bbox_ratios, dtype=np.float32),
        valid=np.array(valid, dtype=np.bool_),
        meta=json.dumps(meta, ensure_ascii=False),
    )

    if save_csv:
        import pandas as pd
        cols = {}
        for j in range(33):
            cols[f"x{j}"] = lm_x[:, j]
            cols[f"y{j}"] = lm_y[:, j]
            cols[f"v{j}"] = lm_v[:, j]
        df = pd.DataFrame({
            "frame": frames,
            "t_ms": t_ms,
            "vis_mean": vis_mean,
            "bbox_ratio": bbox_ratios,
            "valid": valid,
            **cols
        })
        csv_path = out_npz.with_suffix(".csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    return out_npz

# ---------------------------
# 3) CLI
# ---------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Mediapipe Pose time-series from video.")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="results/keypoints/*.npz")
    p.add_argument("--csv", action="store_true", help="CSV도 함께 저장")
    p.add_argument("--every", type=int, default=1, help="프레임 스킵 간격(1=모든 프레임)")
    p.add_argument("--resize", type=int, default=None, help="리사이즈 폭(px). 작게 설정하면 속도↑")
    p.add_argument("--complexity", type=int, default=1, choices=[0,1,2])
    p.add_argument("--autorotate", action="store_true")  # 자리만
    p.add_argument("--rotate", type=int, default=0, choices=[0,90,180,270])
    p.add_argument("--flip", action="store_true")
    args = p.parse_args()

    out = probe_pose(
        args.video,
        out_npz=args.out,
        save_csv=args.csv,
        every=max(1, args.every),
        resize_width=args.resize,
        model_complexity=args.complexity,
        autorotate=args.autorotate,
        rotate=args.rotate,
        flip=args.flip,
    )
    print(f"[saved] {out}")
