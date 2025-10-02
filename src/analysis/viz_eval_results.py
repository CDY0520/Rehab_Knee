# -*- coding: utf-8 -*-
"""
파일명: src/analysis/viz_eval_results.py

설명:
  - 평가 결과 CSV/NPZ 시각화: F1, MAE, TP deltas, 타임라인.
  - 입력 경로는 고정 디렉토리 규칙 사용:
      · results/experiment/events_eval_summary.csv
      · results/experiment/events_eval_pairs_tp.csv
      · results/keypoints/{...}.npz → NPZ_MAP에서 video_id 매핑
      · results/gt/{video_id}_{side}.csv
      · results/experiment/pred_{video_id}.csv
  - 타임라인:
      · 왼쪽 y축: Heel_y, Toe_y (아래=1, 위=0)
      · 오른쪽 y축: 무릎 각도(위=180°, 아래로 각도 작아짐)
      · 하단 영역(축 밖)에 GT/Pred 라벨 표시
출력:
  · results/experiment/viz_f1.png
  · results/experiment/viz_mae.png
  · results/experiment/delta_stats_tp.csv
  · results/experiment/viz_delta_box_{video_id}.png
  · results/experiment/viz_timeline_{video_id}_{side}.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms as mtrans

# ───────────────────────────────
# 고정 경로
EXP_DIR = Path("results/experiment"); EXP_DIR.mkdir(parents=True, exist_ok=True)
GT_DIR  = Path("results/gt")
KEY_DIR = Path("results/keypoints")

SUMMARY_CSV  = EXP_DIR / "events_eval_summary.csv"
PAIRS_TP_CSV = EXP_DIR / "events_eval_pairs_tp.csv"
PRED_CSV_TPL = "pred_{vid}.csv"

# video_id → npz 경로 매핑 (필요에 맞게 추가)
NPZ_MAP = {
    "normal":   KEY_DIR / "sample_walk_normal.npz",
    "hyperext": KEY_DIR / "sample_walk3.npz",
}

# ───────────────────────────────
# 1) F1 막대 그래프
def plot_f1():
    if not SUMMARY_CSV.exists(): return
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["event"] != "ALL"].copy()
    df["F1"] = pd.to_numeric(df["F1"], errors="coerce")
    plt.figure(figsize=(6,4))
    plt.bar(df["event"], df["F1"], color="tab:blue")
    plt.ylim(0,1.05)
    plt.ylabel("F1 score"); plt.title("Event detection F1 (summary)")
    for i,v in enumerate(df["F1"]):
        if pd.notna(v): plt.text(i, v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(EXP_DIR/"viz_f1.png", dpi=150); plt.close()

# 2) MAE 막대 그래프
def plot_mae():
    if not SUMMARY_CSV.exists(): return
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["event"] != "ALL"].copy()
    mae = pd.to_numeric(df["MAE(ms)"], errors="coerce")
    plt.figure(figsize=(6,4))
    plt.bar(df["event"], mae, color="tab:orange")
    plt.ylabel("MAE (ms)"); plt.title("Event timing error (MAE)")
    for i,v in enumerate(mae):
        if pd.notna(v): plt.text(i, v+5, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(EXP_DIR/"viz_mae.png", dpi=150); plt.close()

# 3) TP(delta_ms) 통계·박스플롯
def analyze_tp_deltas():
    if not PAIRS_TP_CSV.exists(): return
    df = pd.read_csv(PAIRS_TP_CSV)
    if "delta_ms" not in df.columns or df.empty: return
    df["delta_ms"] = pd.to_numeric(df["delta_ms"], errors="coerce")
    df = df.dropna(subset=["delta_ms"])
    if df.empty: return

    def p95(x): return np.percentile(x, 95)
    stats = (df.groupby(["video_id","side","event"])["delta_ms"]
               .agg(count="count", mean="mean", median="median", p95=p95, std="std")
               .reset_index().sort_values(["video_id","side","event"]))
    stats.to_csv(EXP_DIR/"delta_stats_tp.csv", index=False, encoding="utf-8-sig")

    for vid, g in df.groupby("video_id"):
        plt.figure(figsize=(8,4))
        g = g.copy(); g["cat"] = g["event"].astype(str)+"_"+g["side"].astype(str)
        order = sorted(g["cat"].unique())
        data = [g[g["cat"]==c]["delta_ms"].values for c in order]
        plt.boxplot(data, labels=order, showfliers=False)
        plt.axhline(0, ls="--", color="gray")
        plt.ylabel("delta_ms (Pred - GT)")
        plt.title(f"Timing error (TP only) - {vid}")
        plt.tight_layout(); plt.savefig(EXP_DIR/f"viz_delta_box_{vid}.png", dpi=150); plt.close()

# ─────────────────────────────────────────────
# 4) 타임라인 오버레이 (timeline.py 스타일)
IDX = dict(
    L_HIP=23, R_HIP=24,
    L_KNEE=25, R_KNEE=26,
    L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30,
    L_TOE=31, R_TOE=32,
)
def angle_abc(a, b, c):
    ba = a - b; bc = c - b
    nba = np.linalg.norm(ba, axis=-1)+1e-8
    nbc = np.linalg.norm(bc, axis=-1)+1e-8
    cosv = np.clip(np.sum(ba*bc, axis=-1)/(nba*nbc), -1,1)
    return np.degrees(np.arccos(cosv))

def plot_timeline_overlay(video_id: str, side: str):
    npz_path = NPZ_MAP.get(video_id)
    if npz_path is None or not npz_path.exists(): return

    gt_csv   = GT_DIR / f"{video_id}_{side}.csv"
    pred_csv = EXP_DIR / PRED_CSV_TPL.format(vid=video_id)

    d = np.load(npz_path, allow_pickle=True)
    lx, ly, t_ms = d["lm_x"].astype(float), d["lm_y"].astype(float), d["t_ms"].astype(float)

    IDX = dict(L_HIP=23,R_HIP=24,L_KNEE=25,R_KNEE=26,L_ANK=27,R_ANK=28,L_HEEL=29,R_HEEL=30,L_TOE=31,R_TOE=32)
    s = side.upper()
    i_heel, i_toe = IDX[f"{s}_HEEL"], IDX[f"{s}_TOE"]
    i_knee, i_hip, i_ank = IDX[f"{s}_KNEE"], IDX[f"{s}_HIP"], IDX[f"{s}_ANK"]

    heel_y = ly[:, i_heel]; toe_y = ly[:, i_toe]
    hip  = np.stack([lx[:,i_hip],  ly[:,i_hip]],  axis=1)
    knee = np.stack([lx[:,i_knee], ly[:,i_knee]], axis=1)
    ank  = np.stack([lx[:,i_ank],  ly[:,i_ank]],  axis=1)
    def angle_abc(a,b,c):
        ba=a-b; bc=c-b
        cos=np.clip(np.sum(ba*bc,axis=-1)/(np.linalg.norm(ba,axis=-1)*np.linalg.norm(bc,axis=-1)+1e-8),-1,1)
        return np.degrees(np.arccos(cos))
    knee_ang = angle_abc(hip,knee,ank)

    gt = pd.read_csv(gt_csv)   if gt_csv.exists()   else pd.DataFrame(columns=["event","time_ms","side"])
    pr = pd.read_csv(pred_csv) if pred_csv.exists() else pd.DataFrame(columns=["event","time_ms","side"])
    if not gt.empty and "side" in gt: gt = gt[gt["side"]==side]
    if not pr.empty and "side" in pr: pr = pr[pr["side"]==side]

    plt.figure(figsize=(12,4))
    ax = plt.gca()
    # 왼쪽 축: y좌표
    ln1, = ax.plot(t_ms, heel_y, label="Heel_y", lw=1.2, color="tab:blue")
    ln2, = ax.plot(t_ms, toe_y,  label="Toe_y",  lw=1.2, color="tab:orange")
    ax.set_ylim(1.0, 0.0)
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("y-coordinate")
    ax.grid(True, alpha=0.25)

    # 오른쪽 축: 무릎 각도(도)
    ax2 = ax.twinx()
    ln3, = ax2.plot(t_ms, knee_ang, label="Knee angle (deg)",
                    lw=1.6, linestyle="-", color="green")
    ax2.set_ylabel("Knee angle (deg)")

    # 각도 축 자동 범위(과도한 아웃라이어 방지용 백분위 기반)
    k_lo = np.nanpercentile(knee_ang, 1)
    k_hi = np.nanpercentile(knee_ang, 99)
    if np.isfinite(k_lo) and np.isfinite(k_hi) and k_hi - k_lo > 1e-3:
        ax2.set_ylim(k_lo - 3, k_hi + 3)

    # 하단 내부에 라벨 찍기
    y0 = 0.90   # GT 점 y
    y1 = 0.95   # Pred 점 y (GT보다 약간 아래쪽으로)
    sc_gt = None
    for i, r in enumerate(gt.itertuples()):
        if i == 0:
            sc_gt = ax.scatter([r.time_ms], [y0], s=36, c="tab:blue", marker="o", zorder=6, label="GT")
        else:
            ax.scatter([r.time_ms], [y0], s=36, c="tab:blue", marker="o", zorder=6)
        ax.text(r.time_ms, y0+0.02, str(r.event), fontsize=9, ha="center", va="bottom", color="tab:red", zorder=6)

    sc_pr = None
    if not pr.empty:
        for i, r in enumerate(pr.itertuples()):
            if i == 0:
                sc_pr = ax.scatter([r.time_ms], [y1], s=40, c="tab:red", marker="x", zorder=6, label="Pred")
            else:
                ax.scatter([r.time_ms], [y1], s=40, c="tab:red", marker="x", zorder=6)
            ax.text(r.time_ms, y1-0.02, str(r.event), fontsize=9, ha="center", va="top", color="tab:red", zorder=6)

    # 범례
    handles = [ln1, ln2, ln3] + ([sc_gt] if sc_gt is not None else []) + ([sc_pr] if sc_pr is not None else [])
    labels  = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", fontsize=9, frameon=True, ncol=min(4,len(handles)))

    ax.set_title(f"Timeline overlay ({'LEFT' if side.upper()=='L' else 'RIGHT'})")
    plt.tight_layout()
    out = EXP_DIR / f"viz_timeline_{video_id}_{side}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    print("[saved]", out)

# ───────────────────────────────
def main():
    plot_f1()
    plot_mae()
    analyze_tp_deltas()

    # pairs_tp.csv의 video_id, side 조합으로 자동 생성
    if PAIRS_TP_CSV.exists():
        pairs = pd.read_csv(PAIRS_TP_CSV)
        if not pairs.empty and {"video_id","side"}.issubset(pairs.columns):
            for (vid, side), _ in pairs.groupby(["video_id", "side"]):
                if vid in NPZ_MAP:
                    plot_timeline_overlay(str(vid), str(side))
        else:
            # 폴백: 기존 기본 예시
            plot_timeline_overlay("normal", "L")
            plot_timeline_overlay("hyperext", "L")
    else:
        plot_timeline_overlay("normal", "L")
        plot_timeline_overlay("hyperext", "L")

if __name__ == "__main__":
    main()
