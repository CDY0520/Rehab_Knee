"""
파일명: src/pose_probe.py

설명:
  - 입력 영상에서 MediaPipe Pose 33 랜드마크(x,y,visibility)를 추출해 시계열(.npz 기본, .csv 옵션)로 저장한다.
  - (포즈 정확도 평가) 라벨 있는 비디오일 때 공식 지표 4개를 산출한다:
      · PCKh@0.5 (하지 관절별/평균)
      · PCKh-AUC(0.1–0.5)
      · OKS-AP50, OKS-mAP(0.50:0.95)
      · Acceleration Error (px/frame^2)
  - 품질 요약 CSV 저장: results/reports/<video>_pose_quality.csv
  - (신규) GT 템플릿 CSV 자동 생성 옵션: results/gt/gt_<video>.csv

블록 구성:
  0) 임포트/상수(매핑·시그마)
  1) 경로 유틸(resolve, _find_gt_csv)
  2) 평가 유틸(OKS, PCKh, Accel)
  3) GT 로더(x0..x32, y0..y32[, v0..v32])
  4) 핵심 추출(probe_pose):
     · MediaPipe 추정 → (x,y,v) 누적
     · (선택) GT와 비교해 4개 지표 계산
     · .npz/.csv(키포인트) 저장, (선택) GT 템플릿 CSV 저장, 품질 요약 CSV 저장
  5) CLI

사용 예:
  포즈만           : python src/pose_probe.py --video data/samples/walk.mp4 --csv
  GT 템플릿 생성    : python src/pose_probe.py --video data/samples/walk.mp4 --make-gt
  품질 평가(수정 후) : python src/pose_probe.py --video data/samples/walk.mp4 --gt-csv results/gt/gt_walk.csv
"""

from __future__ import annotations
import json
from pathlib import Path
import argparse
import cv2
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

# ──────────────────────────────────────────────────────────────────────────────
# 경로·출력 디렉토리
# ──────────────────────────────────────────────────────────────────────────────
KEY_DIR = Path("results") / "keypoints"
REP_DIR = Path("results") / "reports"
GT_DIR  = Path("results") / "gt"
KEY_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [
    PROJECT_ROOT / "data" / "samples",
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "results" / "raw",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "results" / "gt",   # GT 자동 탐색 포함
    PROJECT_ROOT,
]

def resolve_path(p: str) -> Path:
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
    raise FileNotFoundError(f"파일을 찾지 못함: {p}")

def _find_gt_csv(video_path: Path, gt_csv_opt: str | None) -> Path | None:
    # 1) 명시 경로
    if gt_csv_opt:
        p = Path(gt_csv_opt).expanduser()
        if p.exists():
            print(f"[info] GT from --gt-csv: {p}")
            return p
        # 이름만 주어진 경우 탐색
        for d in SEARCH_DIRS:
            for c in d.rglob(p.name):
                print(f"[info] GT matched by name in {d}: {c}")
                return c

    # 2) 스템 기반 후보
    stem = video_path.stem
    name_candidates = [
        f"gt_{stem}.csv", f"{stem}_gt.csv", f"{stem}.gt.csv", f"{stem}.csv"
    ]
    # 같은 폴더
    for n in name_candidates:
        c = video_path.with_name(n)
        if c.exists():
            print(f"[info] auto GT detected (same dir): {c}")
            return c
    # SEARCH_DIRS
    for d in SEARCH_DIRS:
        for n in name_candidates:
            cand = d / n
            if cand.exists():
                print(f"[info] auto GT detected (SEARCH_DIRS): {cand}")
                return cand
        hits = list(d.rglob(f"*{stem}*gt*.csv"))
        if hits:
            print(f"[info] auto GT detected (recursive): {hits[0]}")
            return hits[0]

    return None

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe
# ──────────────────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
except ImportError as e:
    raise ImportError("mediapipe가 필요합니다. pip install mediapipe opencv-python numpy pandas") from e

# ──────────────────────────────────────────────────────────────────────────────
# COCO 17 → MediaPipe 33 매핑 및 시그마(공식)
# ──────────────────────────────────────────────────────────────────────────────
COCO_SIGMAS = np.array([0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089], dtype=np.float32)
COCO_TO_MP = np.array([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28], dtype=np.int32)
LOWER_COCO = np.array([11,12,13,14,15,16], dtype=np.int32)  # hips, knees, ankles
HEAD_MP = np.array([0,2,3,4,5,6,7,8], dtype=np.int32)       # nose, eyes, ears

# ──────────────────────────────────────────────────────────────────────────────
# 유틸: OKS, PCKh, Accel
# ──────────────────────────────────────────────────────────────────────────────
def _to_px(x: np.ndarray, y: np.ndarray, w: int, h: int):
    return x * float(w), y * float(h)

def _bbox_area_px(x_px: np.ndarray, y_px: np.ndarray, vis: np.ndarray | None):
    m = np.ones_like(x_px, dtype=bool) if vis is None else (vis > 0)
    if not np.any(m):
        return 1.0
    xs = x_px[m]; ys = y_px[m]
    bw = max(float(xs.max() - xs.min()), 1.0)
    bh = max(float(ys.max() - ys.min()), 1.0)
    return bw * bh

def oks_per_frame(pred_xy: np.ndarray, gt_xy: np.ndarray, gt_vis: np.ndarray | None, w: int, h: int):
    pm = pred_xy[COCO_TO_MP]
    gm = gt_xy[COCO_TO_MP]
    vm = (gt_vis[COCO_TO_MP] > 0).astype(bool) if gt_vis is not None else np.ones(17, bool)
    px, py = _to_px(pm[:,0], pm[:,1], w, h)
    gx, gy = _to_px(gm[:,0], gm[:,1], w, h)
    s = _bbox_area_px(gx, gy, vm)
    if s <= 0 or not np.any(vm):
        return np.nan, 0
    d2 = (px - gx)**2 + (py - gy)**2
    k = COCO_SIGMAS
    e = np.exp(-d2 / (2 * (s * (k**2) + 1e-9)))
    oks = float(np.sum(e[vm]) / max(1, int(np.sum(vm))))
    return oks, int(np.sum(vm))

def pckh_frame_binary(pred_xy: np.ndarray, gt_xy: np.ndarray, gt_vis: np.ndarray | None,
                      w: int, h: int, alpha: float = 0.5):
    vm33 = (gt_vis > 0).astype(bool) if gt_vis is not None else np.ones(33, bool)
    if np.any(vm33[HEAD_MP]):
        hx, hy = _to_px(gt_xy[HEAD_MP,0], gt_xy[HEAD_MP,1], w, h)
        head_size = max(float(hx.max()-hx.min()), float(hy.max()-hy.min()))
    else:
        head_size = max(w, h) * 0.1
    head_size = max(head_size, 1.0)
    thr = alpha * head_size

    mp_idx = COCO_TO_MP[LOWER_COCO]
    ok_list = []
    for j in mp_idx:
        if not vm33[j]:
            ok_list.append(False)
            continue
        px, py = _to_px(pred_xy[j,0], pred_xy[j,1], w, h)
        gx, gy = _to_px(gt_xy[j,0], gt_xy[j,1], w, h)
        ok_list.append(bool(np.hypot(px-gx, py-gy) <= thr))
    arr = np.array(ok_list, bool)
    return arr, float(arr.mean()) if arr.size else np.nan

def accel_error(pred_xy_seq: np.ndarray, gt_xy_seq: np.ndarray, fps: float, joints_mp: np.ndarray, w: int, h: int):
    if len(pred_xy_seq) < 3:
        return np.nan
    p = pred_xy_seq[:, joints_mp, :] * np.array([w, h], float)
    g = gt_xy_seq[:, joints_mp, :]   * np.array([w, h], float)
    ap = p[2:] - 2*p[1:-1] + p[:-2]
    ag = g[2:] - 2*g[1:-1] + g[:-2]
    diff = ap - ag
    return float(np.mean(np.linalg.norm(diff, axis=2)))

# ──────────────────────────────────────────────────────────────────────────────
# GT 로더 및 저장
# ──────────────────────────────────────────────────────────────────────────────
def load_gt_csv(gt_path: str | Path):
    if pd is None:
        raise RuntimeError("pandas 필요. pip install pandas")
    df = pd.read_csv(gt_path)
    xs = []; ys = []; vs = []
    for j in range(33):
        xs.append(df.get(f"x{j}").to_numpy(float))
        ys.append(df.get(f"y{j}").to_numpy(float))
        vcol = df.get(f"v{j}")
        vs.append((vcol.to_numpy(float) if vcol is not None else np.ones_like(xs[-1])))
    X = np.stack(xs, axis=1)
    Y = np.stack(ys, axis=1)
    V = np.stack(vs, axis=1)
    return X, Y, V

def save_gt_csv(path: Path, frames: list[int], t_ms: list[int],
                lm_x: np.ndarray, lm_y: np.ndarray, lm_v: np.ndarray):
    if pd is None:
        print("[warn] pandas 미설치로 GT CSV 저장 불가"); return
    cols = {"frame": frames, "t_ms": t_ms}
    for j in range(33):
        cols[f"x{j}"] = lm_x[:, j]
        cols[f"y{j}"] = lm_y[:, j]
        cols[f"v{j}"] = lm_v[:, j]
    df = pd.DataFrame(cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[saved] GT template: {path}")

# ──────────────────────────────────────────────────────────────────────────────
# 핵심: 포즈 추출 + 품질 지표 계산 + GT 템플릿 생성(옵션)
# ──────────────────────────────────────────────────────────────────────────────
def probe_pose(
    video_path: str,
    out_npz: str | Path | None = None,
    save_csv_keypoints: bool = False,
    every: int = 1,
    resize_width: int | None = None,
    model_complexity: int = 1,
    autorotate: bool = False,   # 자리만
    rotate: int = 0,
    flip: bool = False,
    task_tag: str = "gait",
    gt_csv: str | None = None,      # 정답 CSV: 없거나 못 찾으면 자동 스킵
    make_gt: bool = False,          # 신규: GT 템플릿 생성
    gt_out: str | None = None,      # 신규: GT 저장 경로 지정(미지정 시 results/gt/gt_<stem>.csv)
) -> Path:
    # 입력·비디오
    vpath = resolve_path(video_path)
    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"open failed: {vpath}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
    )

    # 누적
    frames, t_ms = [], []
    lm_x, lm_y, lm_v = [], [], []

    i = 0
    keep = max(1, int(every))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % keep != 0:
            i += 1
            continue

        # 회전·반전·리사이즈
        r = rotate % 360
        if r == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif r == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif r == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if flip:
            frame = cv2.flip(frame, 1)
        if resize_width is not None and 0 < resize_width < frame.shape[1]:
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
        else:
            x = np.full(33, np.nan, np.float32)
            y = np.full(33, np.nan, np.float32)
            v = np.zeros(33, np.float32)

        lm_x.append(x); lm_y.append(y); lm_v.append(v)
        frames.append(i)
        t_ms.append(int((i / fps) * 1000))
        i += 1

    cap.release(); pose.close()

    # 배열화
    lm_x = np.stack(lm_x, axis=0) if lm_x else np.zeros((0, 33), np.float32)
    lm_y = np.stack(lm_y, axis=0) if lm_y else np.zeros((0, 33), np.float32)
    lm_v = np.stack(lm_v, axis=0) if lm_v else np.zeros((0, 33), np.float32)

    # 메타
    if resize_width and w0 > 0 and h0 > 0:
        out_w = resize_width
        out_h = int(round((resize_width / w0) * h0))
    else:
        out_w = w0
        out_h = h0

    meta = {
        "video": str(vpath),
        "fps": fps,
        "frames_total": total,
        "frames_kept": len(frames),
        "width": out_w,
        "height": out_h,
        "every": keep,
        "rotate": rotate, "flip": flip, "resize_width": resize_width,
        "model_complexity": model_complexity,
        "task": task_tag,
        "metrics": ["PCKh@0.5", "PCKh-AUC(0.1-0.5)", "OKS-AP50", "OKS-mAP", "Accel(px/frame^2)"]
    }

    # 저장(.npz)
    out_npz = Path(out_npz) if out_npz else (KEY_DIR / (vpath.stem + ".npz"))
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        lm_x=lm_x, lm_y=lm_y, lm_v=lm_v,
        frames=np.asarray(frames, np.int32),
        t_ms=np.asarray(t_ms, np.int32),
        valid=np.asarray(~np.isnan(lm_x[:, 0]), np.bool_),
        meta=json.dumps(meta, ensure_ascii=False),
    )

    # 옵션 CSV(키포인트)
    if save_csv_keypoints and pd is not None:
        cols = {}
        for j in range(33):
            cols[f"x{j}"] = lm_x[:, j]
            cols[f"y{j}"] = lm_y[:, j]
            cols[f"v{j}"] = lm_v[:, j]
        df = pd.DataFrame({"frame": frames, "t_ms": t_ms, **cols})
        df.to_csv(out_npz.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    # GT 템플릿 저장(옵션)
    if make_gt:
        gt_save_path = Path(gt_out).expanduser() if gt_out else (GT_DIR / f"gt_{vpath.stem}.csv")
        save_gt_csv(gt_save_path, frames, t_ms, lm_x, lm_y, lm_v)
        print("[note] 방금 생성한 GT 템플릿은 예측값 복사본입니다. 수동 수정 후 평가에 사용하세요.")

    # ─ 품질 지표 계산: GT 자동 탐색
    gt_path = _find_gt_csv(vpath, gt_csv)

    if gt_path is not None and pd is not None:
        GTx, GTy, GTv = load_gt_csv(gt_path)
        T = min(len(frames), GTx.shape[0])
        lm_xy = np.stack([lm_x[:T], lm_y[:T]], axis=-1)  # (T,33,2)
        gt_xy = np.stack([GTx[:T], GTy[:T]], axis=-1)    # (T,33,2)
        gt_v  = GTv[:T]
        W, H = out_w, out_h

        # OKS per-frame → AP50/mAP
        oks_vals = []
        for t in range(T):
            oks_t, _ = oks_per_frame(lm_xy[t], gt_xy[t], gt_v[t], W, H)
            if not np.isnan(oks_t):
                oks_vals.append(oks_t)
        oks_vals = np.array(oks_vals, float)
        Nf = len(oks_vals)
        def ap_at(th): return float(np.mean(oks_vals >= th)) if Nf > 0 else np.nan
        AP50 = ap_at(0.50)
        thresholds = np.arange(0.50, 0.95 + 1e-6, 0.05)
        mAP = float(np.nanmean([ap_at(t) for t in thresholds])) if Nf > 0 else np.nan

        # PCKh@0.5 (하지)
        pckh_bools = []
        for t in range(T):
            okj, _ = pckh_frame_binary(lm_xy[t], gt_xy[t], gt_v[t], W, H, alpha=0.5)
            pckh_bools.append(okj)
        joint_names = ["L_HIP", "R_HIP", "L_KNEE", "R_KNEE", "L_ANKLE", "R_ANKLE"]
        pckh_joint = {}
        if len(pckh_bools) and isinstance(pckh_bools[0], np.ndarray):
            B = np.stack([b for b in pckh_bools if isinstance(b, np.ndarray)], axis=0)  # (T,6)
            for j, name in enumerate(joint_names):
                pckh_joint[name] = float(np.mean(B[:, j]))
            PCKh05_mean = float(np.mean(B))
        else:
            for name in joint_names: pckh_joint[name] = np.nan
            PCKh05_mean = np.nan

        # PCKh-AUC(0.1~0.5)
        alphas = np.arange(0.10, 0.50 + 1e-6, 0.05)
        auc_vals = []
        for a in alphas:
            tmp = []
            for t in range(T):
                _, m = pckh_frame_binary(lm_xy[t], gt_xy[t], gt_v[t], W, H, alpha=a)
                if not np.isnan(m):
                    tmp.append(m)
            auc_vals.append(np.mean(tmp) if tmp else np.nan)
        PCKh_AUC = float(np.nanmean(auc_vals)) if auc_vals else np.nan

        # Accel Error (hips/knees/ankles)
        joints_mp = COCO_TO_MP[LOWER_COCO]
        AccelErr = accel_error(lm_xy, gt_xy, fps, joints_mp, W, H)

        # 저장
        out_quality = REP_DIR / f"{vpath.stem}_pose_quality.csv"
        rows = [{
            "video": vpath.name,
            "frames_used": int(T),
            "oks_ap50": AP50,
            "oks_map_50_95": mAP,
            "pckh05_mean_lower": PCKh05_mean,
            "pckh_auc_0.1_0.5": PCKh_AUC,
            "accel_err_px_per_f2": AccelErr,
            **{f"pckh05_{k.lower()}": v for k, v in pckh_joint.items()},
        }]
        if pd is not None:
            pd.DataFrame(rows).to_csv(out_quality, index=False, encoding="utf-8-sig")
            print(f"[saved] {out_quality}")
    else:
        if gt_csv:
            print(f"[warn] GT CSV not found: {gt_csv}")
        print("[info] GT 미제공/미탐지. 포즈 키포인트만 저장.")

    return out_npz

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="MediaPipe Pose extraction + pose-quality metrics (with GT) + GT template export.")
    p.add_argument("--video", required=True)
    p.add_argument("--out", default=None, help="결과 npz 경로(기본: results/keypoints/<video>.npz)")
    p.add_argument("--csv", action="store_true", help="키포인트 CSV도 저장")
    p.add_argument("--every", type=int, default=1)
    p.add_argument("--resize", type=int, default=None, help="리사이즈 폭(px)")
    p.add_argument("--complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--autorotate", action="store_true")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270])
    p.add_argument("--flip", action="store_true")
    p.add_argument("--task", choices=["gait","sts"], default="gait")
    p.add_argument("--gt-csv", default=None, help="정답 키포인트 CSV 경로(x0..x32,y0..y32[,v0..v32])")
    p.add_argument("--make-gt", action="store_true", help="예측값으로 GT 템플릿 CSV 생성")
    p.add_argument("--gt-out", default=None, help="GT 템플릿 저장 경로(기본: results/gt/gt_<video>.csv)")

    a = p.parse_args()
    out = probe_pose(
        video_path=a.video,
        out_npz=a.out,
        save_csv_keypoints=a.csv,
        every=max(1, a.every),
        resize_width=a.resize,
        model_complexity=a.complexity,
        autorotate=a.autorotate,
        rotate=a.rotate,
        flip=a.flip,
        task_tag=a.task,
        gt_csv=a.gt_csv,
        make_gt=a.make_gt,
        gt_out=a.gt_out,
    )
    print(f"[saved] {out}")
