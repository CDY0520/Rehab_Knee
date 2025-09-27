# -*- coding: utf-8 -*-
"""
파일명: src/analysis/timeline_knee_x.py

기능:
  - 포즈 npz(lm_x, lm_y, t_ms, valid)와 GT csv(event,time_ms)를 입력받아
    (1) Knee angle(도), (2) Knee_x(0~1), (3) GT 포인트만 표시하는 타임라인을 저장.
  - --normal-npz 제공 시 정상보행의 Knee_x를 환자 그래프에 함께 오버레이.

사용 예:
  python src/analysis/timeline_knee_x.py \
    --pose results/keypoints/hyperext_L.npz --gt results/gt/hyperext_L.csv \
    --side L --outdir results/plots \
    --normal-npz results/keypoints/normal_L.npz
"""

# 0) 라이브러리 임포트
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Mediapipe 인덱스 정의
IDX = dict(
    L_HIP=23, R_HIP=24,
    L_KNEE=25, R_KNEE=26,
    L_ANK=27, R_ANK=28,
)

# 2) 각도 계산 유틸
def angle_abc(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba, axis=-1) + 1e-8
    nbc = np.linalg.norm(bc, axis=-1) + 1e-8
    cosv = np.clip(np.sum(ba*bc, axis=-1)/(nba*nbc), -1, 1)
    return np.degrees(np.arccos(cosv))

# 3) npz 로드
def load_pose_npz(path_npz):
    d = np.load(path_npz, allow_pickle=True)
    lm_x = d["lm_x"].astype(float); lm_y = d["lm_y"].astype(float)
    if "t_ms" in d:
        t_ms = d["t_ms"].astype(float)
    else:
        fps = 30.0
        if "meta" in d and isinstance(d["meta"].item(), dict):
            fps = float(d["meta"].item().get("fps", 30.0))
        t_ms = np.arange(lm_x.shape[0]) / fps * 1000.0
    valid = d["valid"].astype(bool) if "valid" in d else np.ones(lm_x.shape[0], bool)
    return lm_x, lm_y, t_ms, valid

# 4) GT 로드
def load_gt_csv(path_csv):
    if not path_csv or not os.path.isfile(path_csv):
        return None
    df = pd.read_csv(path_csv)
    df.columns = [c.strip() for c in df.columns]
    if "event" not in df.columns or "time_ms" not in df.columns:
        raise ValueError("GT csv에 event,time_ms 필요")
    out = df[["event","time_ms"]].copy()
    out["event"] = out["event"].astype(str).str.strip().str.upper()
    out["time_ms"] = pd.to_numeric(out["time_ms"], errors="coerce")
    out = out.dropna(subset=["time_ms"]).reset_index(drop=True)
    return out

# 5) 시계열 추출: knee_angle, knee_x
def extract_knee_series(lm_x, lm_y, valid, side="L"):
    s = side.upper()
    i_knee = IDX[f"{s}_KNEE"]; i_hip = IDX[f"{s}_HIP"]; i_ank = IDX[f"{s}_ANK"]
    knee_x = lm_x[:, i_knee].astype(float).copy()

    hip  = np.stack([lm_x[:, i_hip],  lm_y[:, i_hip]],  axis=1).astype(float)
    knee = np.stack([lm_x[:, i_knee], lm_y[:, i_knee]], axis=1).astype(float)
    ank  = np.stack([lm_x[:, i_ank],  lm_y[:, i_ank]],  axis=1).astype(float)
    knee_angle_deg = angle_abc(hip, knee, ank)

    knee_x[~valid] = np.nan
    knee_angle_deg[~valid] = np.nan
    return knee_angle_deg, knee_x

# 6) 플롯: 각도(좌), knee_x(우, 아래=1), GT 포인트
def plot_knee_only(t_ms, knee_ang_deg, knee_x, gt_df, side, save_path,
                   normal_knee_x=None):
    plt.figure(figsize=(14, 4.8))
    ax = plt.gca()

    # 좌측 y축: 무릎 각도
    ln1, = ax.plot(t_ms, knee_ang_deg, lw=1.8, color="green", label="Knee angle(GR)")
    ax.set_title(f"Knee timeline ({'LEFT' if side=='L' else 'RIGHT'})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("Knee angle (deg)")
    ax.grid(True, alpha=0.25)
    k_lo = np.nanpercentile(knee_ang_deg, 1); k_hi = np.nanpercentile(knee_ang_deg, 99)
    if np.isfinite(k_lo) and np.isfinite(k_hi) and k_hi-k_lo>1e-3:
        ax.set_ylim(k_lo-3, k_hi+3)

    # 우측 y축: knee_x (0~1, 아래=1로 고정)
    ax2 = ax.twinx()
    ln2, = ax2.plot(t_ms, knee_x, lw=1.6, linestyle="--", label="Knee_x(GR)")
    if normal_knee_x is not None:
        ln3, = ax2.plot(t_ms, normal_knee_x, lw=1.2, linestyle=":", label="Knee_x (normal)")
    ax2.set_ylabel("Knee x-coordinate")
    ax2.set_ylim(1.0, 0.0)  # 아래=1, 위=0

    # GT 포인트(오른쪽 축 기준, 아래쪽 근처에 라벨)
    sc_gt = None
    if gt_df is not None and len(gt_df) > 0:
        y0 = 0.95  # 아래쪽에 찍기(우측 축 기준)
        for k, (t, lab) in enumerate(zip(gt_df["time_ms"].values, gt_df["event"].values)):
            if k == 0:
                sc_gt = ax2.scatter([t], [y0], s=40, c="tab:blue", marker="o", zorder=6, label="GT(GR)")
            else:
                ax2.scatter([t], [y0], s=40, c="tab:blue", marker="o", zorder=6)
            ax2.text(t, y0-0.04, lab, fontsize=9, ha="center", va="top", color="tab:red", zorder=6)

    # 범례
    handles = [ln1, ln2] + ([ln3] if normal_knee_x is not None else []) + ([sc_gt] if sc_gt is not None else [])
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", fontsize=9, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

# 7) 메인
def main():
    ap = argparse.ArgumentParser(description="Knee-only timeline plotter")
    ap.add_argument("--pose", "--npz", dest="npz", required=True, help="분석 대상 npz")
    ap.add_argument("--gt", default=None, help="GT csv 경로")
    ap.add_argument("--side", default="L", choices=["L","R"])
    ap.add_argument("--outdir", default="results/plots")
    ap.add_argument("--normal-npz", default=None, help="정상보행 npz(옵션). knee_x만 오버레이")
    ap.add_argument("--normal-side", default=None, help="정상보행 측(기본: --side와 동일)")
    args = ap.parse_args()

    # 대상 로드
    lm_x, lm_y, t_ms, valid = load_pose_npz(args.npz)
    knee_ang, knee_x = extract_knee_series(lm_x, lm_y, valid, side=args.side)
    gt_df = load_gt_csv(args.gt) if args.gt else None

    # 정상보행 knee_x 준비
    normal_knee_x = None
    if args.normal_npz:
        n_lx, n_ly, n_t, n_valid = load_pose_npz(args.normal_npz)
        n_side = args.normal_side if args.normal_side else args.side
        _, n_knee_x = extract_knee_series(n_lx, n_ly, n_valid, side=n_side)
        # 시간 길이 다르면 앞뒤 최소 길이에 맞춰 자르기
        L = min(len(t_ms), len(n_t))
        t_ms = t_ms[:L]; knee_ang = knee_ang[:L]; knee_x = knee_x[:L]; n_knee_x = n_knee_x[:L]
        normal_knee_x = n_knee_x

    # 저장 경로
    base = os.path.splitext(os.path.basename(args.npz))[0]
    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"{base}_{args.side}_knee_only.png")

    # 플롯
    saved = plot_knee_only(t_ms, knee_ang, knee_x, gt_df, args.side, save_path,
                           normal_knee_x=normal_knee_x)
    print(f"[saved] {saved}")

if __name__ == "__main__":
    main()
