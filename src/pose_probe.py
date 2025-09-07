"""
파일명: src/pose_probe.py
설명:
  - 입력 영상에서 Mediapipe Pose로 프레임별 33개 랜드마크(x, y, visibility)를 추출해 시계열(.npz 기본, .csv 옵션)로 저장한다.
  - 보행(Gait)과 STS(앉았다 일어서기) 모두 공용으로 사용한다. 이벤트 검출은 하지 않고, 후속(events.py) 단계의 입력을 생성한다.
  - 품질지표(vis_mean, n_visible, bbox_ratio, jitter, quality_ok)를 함께 산출한다.

블록 구성:
  0) 임포트/상수
  1) 유틸(경로 탐색, 회전/반전, bbox 비율)
  2) 핵심 추출(probe_pose):
     · 프레임 루프 → (x,y,v) 누적
     · 품질지표 계산 및 저장
  3) 저장(np.savez_compressed, 옵션 csv)
  4) CLI:
     --video, --out, --csv, --every, --resize, --complexity,
     --autorotate, --rotate, --flip,
     --task {gait,sts},            # 메타에 기록(후속 분석 구분용)
     --q-vis-min, --q-bbox-min, --q-bbox-max, --q-jitter-max

사용 예:
  보행: python src/pose_probe.py --video data/samples/sample_walk.mp4 --task gait --csv
  STS : python src/pose_probe.py --video data/samples/sample_sts.mp4  --task sts  --csv
"""

from __future__ import annotations
import json
from pathlib import Path
import argparse
import cv2
import numpy as np

# ── Mediapipe 의존성
try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe가 필요합니다. pip install mediapipe opencv-python numpy") from e

# ── 출력 디렉토리
KEY_DIR = Path("results") / "keypoints"
KEY_DIR.mkdir(parents=True, exist_ok=True)

# ── 프로젝트 루트 및 탐색 경로(파일명만 줘도 검색)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "samples",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "results" / "raw",
    PROJECT_ROOT / "results",
    PROJECT_ROOT,
]

def resolve_video_path(p: str) -> Path:
    q = Path(p).expanduser()
    if q.exists():
        return q
    name = Path(p).name
    for d in SEARCH_DIRS:
        cand = d / name
        if cand.exists():
            return cand
    for d in SEARCH_DIRS:
        hits = list(d.rglob(name))
        if hits:
            return hits[0]
    raise FileNotFoundError(f"영상 파일을 찾지 못함: {p}\n검색 폴더: " + ", ".join(str(d) for d in SEARCH_DIRS))

def _apply_rotate_flip(frame, rotate_deg: int = 0, flip: bool = False):
    r = rotate_deg % 360
    if r == 90:   frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif r == 180: frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif r == 270: frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if flip:
        frame = cv2.flip(frame, 1)
    return frame

def _bbox_ratio(xy: np.ndarray, w: int, h: int) -> float:
    xs = xy[:, 0] * w
    ys = xy[:, 1] * h
    bw = max(float(xs.max() - xs.min()), 1.0)
    bh = max(float(ys.max() - ys.min()), 1.0)
    denom = max(float(w * h), 1.0)
    return (bw * bh) / denom

# ── 핵심: 포즈 추출
def probe_pose(
    video_path: str,
    out_npz: str | Path | None = None,
    save_csv: bool = False,
    every: int = 1,
    resize_width: int | None = None,
    model_complexity: int = 1,
    autorotate: bool = False,   # 자리만
    rotate: int = 0,
    flip: bool = False,
    task_tag: str = "gait",     # {"gait","sts"} 메타 표기용
    q_vis_min: float = 0.6,
    q_bbox_min: float = 0.01,
    q_bbox_max: float = 0.6,
    q_jitter_max: float = 0.02,
) -> Path:
    path = resolve_video_path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    frames, t_ms = [], []
    vis_mean, bbox_ratios, valid = [], [], []
    lm_x, lm_y, lm_v = [], [], []

    # 품질 지표
    n_visible, jitter, quality_ok = [], [], []
    KEY_JOINTS = [23, 24, 25, 26, 27, 28, 31, 32]  # hips, knees, ankles, toes
    last_core_xy = None

    i = 0
    keep = max(1, int(every))

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % keep != 0:
            i += 1
            continue

        frame = _apply_rotate_flip(frame, rotate_deg=rotate, flip=flip)

        if resize_width is not None and resize_width > 0 and resize_width < frame.shape[1]:
            scale = resize_width / frame.shape[1]
            nh = max(int(round(frame.shape[0] * scale)), 1)
            frame = cv2.resize(frame, (resize_width, nh), interpolation=cv2.INTER_AREA)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        if res.pose_landmarks:
            lm = res.pose_landmarks.landmark
            x = np.fromiter((p.x for p in lm), dtype=np.float32, count=33)
            y = np.fromiter((p.y for p in lm), dtype=np.float32, count=33)
            v = np.fromiter((p.visibility for p in lm), dtype=np.float32, count=33)

            vmean_cur = float(v.mean())
            xy = np.stack([x, y], axis=1)
            bbr_cur = _bbox_ratio(xy, frame.shape[1], frame.shape[0])

            if last_core_xy is None:
                j_cur = 0.0
            else:
                j_cur = float(np.nanmean(np.sqrt(((x[KEY_JOINTS] - last_core_xy[:, 0]) ** 2) +
                                                 ((y[KEY_JOINTS] - last_core_xy[:, 1]) ** 2))))
            last_core_xy = np.stack([x[KEY_JOINTS], y[KEY_JOINTS]], axis=1)

            q_ok = (vmean_cur >= q_vis_min) and (q_bbox_min <= bbr_cur <= q_bbox_max) and (j_cur <= q_jitter_max)

            lm_x.append(x); lm_y.append(y); lm_v.append(v)
            vis_mean.append(vmean_cur)
            bbox_ratios.append(bbr_cur)
            n_visible.append(int((v >= 0.5).sum()))
            jitter.append(j_cur)
            quality_ok.append(bool(q_ok))
            valid.append(True)
        else:
            lm_x.append(np.full(33, np.nan, np.float32))
            lm_y.append(np.full(33, np.nan, np.float32))
            lm_v.append(np.zeros(33, np.float32))
            vis_mean.append(0.0)
            bbox_ratios.append(0.0)
            n_visible.append(0)
            jitter.append(np.nan)
            quality_ok.append(False)
            valid.append(False)

        frames.append(i)
        t_ms.append(int((i / fps) * 1000))
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
        "every": keep,
        "rotate": rotate,
        "flip": flip,
        "resize_width": resize_width,
        "model_complexity": model_complexity,
        "task": task_tag,  # "gait" 또는 "sts"
        "quality_thresholds": {
            "vis_min": q_vis_min,
            "bbox_min": q_bbox_min,
            "bbox_max": q_bbox_max,
            "jitter_max": q_jitter_max,
        },
    }

    out_npz = Path(out_npz) if out_npz else (KEY_DIR / (path.stem + ".npz"))
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_npz,
        lm_x=lm_x, lm_y=lm_y, lm_v=lm_v,
        frames=np.asarray(frames, np.int32),
        t_ms=np.asarray(t_ms, np.int32),
        vis_mean=np.asarray(vis_mean, np.float32),
        n_visible=np.asarray(n_visible, np.int32),
        bbox_ratio=np.asarray(bbox_ratios, np.float32),
        jitter=np.asarray(jitter, np.float32),
        quality_ok=np.asarray(quality_ok, np.bool_),
        valid=np.asarray(valid, np.bool_),
        meta=json.dumps(meta, ensure_ascii=False),
    )

    # 품질 요약
    q = np.asarray(quality_ok, bool)
    print("=== POSE QUALITY ===")
    print(f"frames={len(frames)} | kept={int(q.sum())} ({100*float(q.mean()):.1f}%)")
    print(f"vis_mean avg={np.nanmean(vis_mean):.3f}  bbox median={np.nanmedian(bbox_ratios):.4f}")
    print(f"jitter median={np.nanmedian(jitter):.4f}")

    if save_csv:
        try:
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
                "n_visible": n_visible,
                "bbox_ratio": bbox_ratios,
                "jitter": jitter,
                "quality_ok": quality_ok,
                "valid": valid,
                **cols
            })
            csv_path = out_npz.with_suffix(".csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        except Exception as e:
            print(f"[warn] CSV 저장 실패: {e}")

    return out_npz

# ── CLI
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Extract Mediapipe Pose time-series for gait/STS.")
    p.add_argument("--video", required=True, help="영상 경로 또는 파일명(자동 탐색)")
    p.add_argument("--out", default=None, help="결과 npz 경로(기본: results/keypoints/<video>.npz)")
    p.add_argument("--csv", action="store_true", help="CSV도 함께 저장")
    p.add_argument("--every", type=int, default=1, help="프레임 스킵 간격(1=모든 프레임)")
    p.add_argument("--resize", type=int, default=None, help="리사이즈 폭(px)")
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2], help="Mediapipe Pose 모델 복잡도")
    p.add_argument("--autorotate", action="store_true")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--flip", action="store_true")
    p.add_argument("--task", choices=["gait", "sts"], default="gait", help="메타에 기록할 분석 과제 태그")

    # 품질 임계
    p.add_argument("--q-vis-min", type=float, default=0.6)
    p.add_argument("--q-bbox-min", type=float, default=0.01)
    p.add_argument("--q-bbox-max", type=float, default=0.6)
    p.add_argument("--q-jitter-max", type=float, default=0.02)

    a = p.parse_args()
    out = probe_pose(
        a.video, out_npz=a.out, save_csv=a.csv, every=max(1, a.every),
        resize_width=a.resize, model_complexity=a.complexity,
        autorotate=a.autorotate, rotate=a.rotate, flip=a.flip,
        task_tag=a.task,
        q_vis_min=a.q_vis_min, q_bbox_min=a.q_bbox_min,
        q_bbox_max=a.q_bbox_max, q_jitter_max=a.q_jitter_max,
    )
    print(f"[saved] {out}")
