"""
파일명: src/analysis/viz_eval_results.py
설명(최신):
  - 평가 결과 CSV/NPZ를 읽어 정확도·오차·타임라인을 시각화한다.
  - 입력:
      · results/experiment/events_eval_summary.csv
      · results/experiment/events_eval_pairs_tp.csv
      · results/keypoints/sample_walk_normal.npz  (normal)
      · results/keypoints/sample_walk.npz         (hyperext)
      · results/gt/{video_id}_{side}.csv
      · results/experiment/pred_{video_id}.csv
  - 출력:
      · results/experiment/viz_f1.png
      · results/experiment/viz_mae.png
      · results/experiment/delta_stats_tp.csv
      · results/experiment/viz_delta_box_{video_id}.png
      · results/experiment/viz_timeline_{video_id}_{side}.png
블록 구성:
  0) 경로/매핑 정의
  1) F1 막대 그래프
  2) MAE 막대 그래프
  3) TP(delta_ms) 통계·박스플롯
  4) 타임라인 오버레이(Heel/Toe y + GT/Pred 이벤트)
  5) main: normal(L), hyperext(R) 기본 렌더
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

# 0) 경로/매핑 정의 --------------------------------------------------------------
EXP_DIR = Path("results/experiment")
GT_DIR = Path("results/gt")
KEY_DIR = Path("results/keypoints")

SUMMARY_CSV   = EXP_DIR / "events_eval_summary.csv"
PAIRS_TP_CSV  = EXP_DIR / "events_eval_pairs_tp.csv"
PRED_CSV_TPL  = "pred_{vid}.csv"

# video_id → npz 경로 매핑 (요청 반영)
NPZ_MAP = {
    "normal":   KEY_DIR / "sample_walk_normal.npz",
    "hyperext": KEY_DIR / "sample_walk4.npz",
}

# ───────────────────────────────────────────────────────────────────────────────
# 1) F1 막대 그래프
def plot_f1():
    if not SUMMARY_CSV.exists():
        print("[skip] not found:", SUMMARY_CSV); return
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["event"] != "ALL"].copy()
    df["F1"] = pd.to_numeric(df["F1"], errors="coerce")
    plt.figure(figsize=(6,4))
    plt.bar(df["event"], df["F1"])
    plt.ylim(0,1.05)
    plt.ylabel("F1 score"); plt.title("Event detection F1 (summary)")
    for i,v in enumerate(df["F1"]):
        if pd.notna(v): plt.text(i, v+0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(EXP_DIR/"viz_f1.png", dpi=150); plt.close()
    print("[saved]", EXP_DIR/"viz_f1.png")

# 2) MAE 막대 그래프
def plot_mae():
    if not SUMMARY_CSV.exists():
        print("[skip] not found:", SUMMARY_CSV); return
    df = pd.read_csv(SUMMARY_CSV)
    df = df[df["event"] != "ALL"].copy()
    mae = pd.to_numeric(df["MAE(ms)"], errors="coerce")
    plt.figure(figsize=(6,4))
    plt.bar(df["event"], mae)
    plt.ylabel("MAE (ms)"); plt.title("Event timing error (MAE)")
    for i,v in enumerate(mae):
        if pd.notna(v): plt.text(i, v+5, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(EXP_DIR/"viz_mae.png", dpi=150); plt.close()
    print("[saved]", EXP_DIR/"viz_mae.png")

# 3) TP(delta_ms) 통계·박스플롯
def analyze_tp_deltas():
    if not PAIRS_TP_CSV.exists():
        print("[skip] not found:", PAIRS_TP_CSV); return
    df = pd.read_csv(PAIRS_TP_CSV)
    if "delta_ms" not in df.columns or df.empty:
        print("[skip] empty TP pairs"); return
    df["delta_ms"] = pd.to_numeric(df["delta_ms"], errors="coerce")
    df = df.dropna(subset=["delta_ms"])
    if df.empty:
        print("[skip] no numeric deltas"); return

    def p95(x): return np.percentile(x, 95)
    stats = (df.groupby(["video_id","side","event"])["delta_ms"]
               .agg(count="count", mean="mean", median="median", p95=p95, std="std")
               .reset_index().sort_values(["video_id","side","event"]))
    stats.to_csv(EXP_DIR/"delta_stats_tp.csv", index=False, encoding="utf-8-sig")
    print("[saved]", EXP_DIR/"delta_stats_tp.csv")

    for vid, g in df.groupby("video_id"):
        plt.figure(figsize=(8,4))
        g = g.copy(); g["cat"] = g["event"].astype(str)+"_"+g["side"].astype(str)
        order = sorted(g["cat"].unique())
        data = [g[g["cat"]==c]["delta_ms"].values for c in order]
        plt.boxplot(data, labels=order, showfliers=False)
        plt.axhline(0, ls="--")
        plt.ylabel("delta_ms (Pred - GT)")
        plt.title(f"Timing error (TP only) - {vid}")
        plt.tight_layout(); plt.savefig(EXP_DIR/f"viz_delta_box_{vid}.png", dpi=150); plt.close()
        print("[saved]", EXP_DIR/f"viz_delta_box_{vid}.png")

# 4) 타임라인 오버레이(Heel/Toe y + GT/Pred)
def _load_gt(video_id: str, side: str) -> pd.DataFrame:
    rows = []
    p = GT_DIR / f"{video_id}_{side}.csv"
    if p.exists():
        g = pd.read_csv(p)
        if not g.empty: rows.append(g[g["side"]==side])
    return pd.concat(rows) if rows else pd.DataFrame(columns=["event","time_ms","side"])

def _load_pred(video_id: str, side: str) -> pd.DataFrame:
    p = EXP_DIR / PRED_CSV_TPL.format(vid=video_id)
    if not p.exists(): return pd.DataFrame(columns=["event","time_ms","side"])
    d = pd.read_csv(p)
    return d[d["side"]==side] if "side" in d.columns else d

def plot_timeline_overlay(video_id: str, side: str):
    npz_path = NPZ_MAP.get(video_id)
    if npz_path is None or not npz_path.exists():
        print("[skip] keypoints not found:", npz_path); return
    d = np.load(npz_path, allow_pickle=True)
    ly, t_ms = d["lm_y"], d["t_ms"]
    heel_y = ly[:, 29] if side=="L" else ly[:, 30]
    toe_y  = ly[:, 31] if side=="L" else ly[:, 32]

    gt = _load_gt(video_id, side)
    pr = _load_pred(video_id, side)

    plt.figure(figsize=(12,4))
    plt.plot(t_ms, heel_y, label="Heel_y")
    plt.plot(t_ms, toe_y,  label="Toe_y")

    # GT = ● 파란색
    for i, r in enumerate(gt.itertuples()):
        lbl = "GT" if i == 0 else None
        plt.scatter(r.time_ms, 0, c="blue", marker="o", label=lbl)
        plt.text(r.time_ms, 0.0, r.event, color="blue", fontsize=8,
                 ha="center", va="top")

    # Pred = × 빨간색
    for i, r in enumerate(pr.itertuples()):
        plbl = "Pred" if i == 0 else None
        plt.scatter(r.time_ms, 0, c="red", marker="x", label=plbl)
        plt.text(r.time_ms, 0.0, r.event, color="red", fontsize=8,
                 ha="center", va="bottom")

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.title(f"Timeline overlay {video_id} {side}")
    plt.xlabel("time (ms)"); plt.ylabel("y-coord (normalized)")
    plt.tight_layout()
    out = EXP_DIR / f"viz_timeline_{video_id}_{side}.png"
    plt.savefig(out, dpi=150); plt.close()
    print("[saved]", out)

# 5) main -----------------------------------------------------------------------
def main():
    plot_f1()
    plot_mae()
    analyze_tp_deltas()
    # 요청: keypoints 파일명 기준 normal(L), hyperext(R) 기본 출력
    plot_timeline_overlay("normal", "L")
    plot_timeline_overlay("hyperext", "R")

if __name__ == "__main__":
    main()
