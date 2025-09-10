"""
파일명: src/analysis/run_qmetrics_threshold.py
목적: data/samples 의 5개 영상을 대상으로
      1) qmetrics 계산 → result/experiment/qmetrics_compare.csv
      2) mRR(성능 유지율) 계산·저장·그래프
      3) TOST(rule-only) 그래프
      4) 임계치(threshold) 타당성(혼동행렬/F1) 평가 + 혼동행렬 그래프
      5) 규칙 위반 지표 디버그 출력(어두운 의복 등 원인 파악)
출력: result/experiment/*.csv, result/plots/*.png
"""

import sys, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

# -----------------------------
# 경로/설정
# -----------------------------
ROOT = Path(".")
sys.path.insert(0, str(ROOT))
from src.qmetrics import compute_qmetrics   # 필수

SAMPLE_DIR = ROOT / "data" / "samples"
OUT_EXP = ROOT / "resultS" / "experiment"
OUT_PLOT = ROOT / "resultS" / "plots"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

# 실험 대상 (3 FAIL + 2 PASS)
EXPECT = {
    "FAIL-Front-Blur": "front_blur_FAIL.mp4",
    "FAIL-Back-Blur" : "back_blur_FAIL.mp4",
    "FAIL-Occlusion" : "45_occlusion_FAIL.mp4",
    "PASS-BrightSide": "side_bright_PASS.mp4",
    "PASS-DarkSide"  : "side_dark_PASS.mp4",
}

BASELINE_LABEL = "PASS-BrightSide"  # 기준
HIGH_BETTER = ["avg_visibility", "visible_ratio"]
LOW_BETTER  = ["jitter_std", "occlusion_rate"]
EQ_BOUNDS   = {"avg_visibility":0.05, "visible_ratio":0.05,
               "jitter_std":0.005, "occlusion_rate":0.15}
THRESHOLDS  = {"avg_visibility":0.80, "visible_ratio":0.85,
               "jitter_std":0.005, "occlusion_rate":0.15}

# -----------------------------
# 1) qmetrics 계산
# -----------------------------
rows = []
for label, fname in EXPECT.items():
    path = SAMPLE_DIR / fname
    if not path.exists():
        print(f"[MISS] {label} -> {fname}")
        continue
    m = compute_qmetrics(str(path), sample_cap=150)
    if "error" in m:
        print(f"[FAIL] {label}: {m['error']}"); continue
    rows.append({
        "label": label, "video": path.name,
        "fps": m.get("fps", np.nan),
        "avg_visibility": m.get("avg_visibility", np.nan),
        "visible_ratio": m.get("visible_ratio", np.nan),
        "occlusion_rate": m.get("occlusion_rate", np.nan),
        "jitter_std": m.get("jitter_std", np.nan),
    })
    print(f"[OK] {label:16s} vis={m.get('avg_visibility',np.nan):.3f} "
          f"vr={m.get('visible_ratio',np.nan):.3f} occ={m.get('occlusion_rate',np.nan):.3f} "
          f"jit={m.get('jitter_std',np.nan):.3f}")

df = pd.DataFrame(rows).sort_values("label")
csv_path = OUT_EXP / "qmetrics_th_compare.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")

# 라벨(파일명에 _FAIL → 1, _PASS → 0)
df["needs_reshoot"] = df["video"].apply(lambda x: 1 if re.search(r"_FAIL", x, re.I) else 0)
lab_csv = OUT_EXP / "qmetrics_th_labeled.csv"
df.to_csv(lab_csv, index=False, encoding="utf-8-sig")

# -----------------------------
# 2) mRR 계산 + 그래프
# -----------------------------
base = df[df["label"]==BASELINE_LABEL].iloc[0]

def mrr_val(row, key):
    b, v = float(base[key]), float(row[key])
    if np.isnan(b) or np.isnan(v) or b==0 or v==0: return np.nan
    return (v/b) if key in HIGH_BETTER else (b/v)

mrr = df[["label"]].copy()
for k in HIGH_BETTER+LOW_BETTER:
    mrr[f"mRR_{k}"] = df.apply(lambda r: mrr_val(r,k), axis=1)
mrr["mRR_mean"] = mrr[[c for c in mrr.columns if c.startswith("mRR_")]].mean(axis=1)
mrr_csv = OUT_EXP / "mrr_th_table.csv"
mrr.to_csv(mrr_csv, index=False, encoding="utf-8-sig")

plt.figure(figsize=(7.5,4))
plt.bar(mrr["label"], mrr["mRR_mean"])
plt.axhline(1.0, linestyle="--")
plt.title(f"Relative Robustness vs. {BASELINE_LABEL}")
plt.ylabel("mRR mean")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_PLOT / "mRR_mean_th.png", dpi=160)
plt.close()

# -----------------------------
# 3) TOST(rule-only) 그래프
# -----------------------------
for metric, bound in EQ_BOUNDS.items():
    sub = []
    for _, r in df.iterrows():
        if r["label"] == BASELINE_LABEL: continue
        delta = float(r[metric] - base[metric])
        eq = abs(delta) <= bound
        sub.append((r["label"], delta, eq))
    if not sub: continue
    labels, deltas, eqs = zip(*sub)
    colors = ["#2ca02c" if e else "#d62728" for e in eqs]
    plt.figure(figsize=(7.5,3))
    plt.bar(labels, deltas, color=colors)
    plt.axhline(+bound, linestyle="--"); plt.axhline(-bound, linestyle="--")
    plt.title(f"TOST delta vs bounds for {metric} (rule-only, ref={BASELINE_LABEL})")
    plt.ylabel("delta (cond - base)")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_PLOT / f"tost_th_{metric}.png", dpi=160)
    plt.close()

# -----------------------------
# 4) 임계치 타당성(혼동행렬/F1) + 그래프
# -----------------------------
def rule_pass(d):
    return (
        (d["avg_visibility"] >= THRESHOLDS["avg_visibility"]) &
        (d["visible_ratio"]  >= THRESHOLDS["visible_ratio"])  &
        (d["jitter_std"]     <= THRESHOLDS["jitter_std"])     &
        (d["occlusion_rate"] <= THRESHOLDS["occlusion_rate"])
    ).astype(int)

y_true = df["needs_reshoot"].values            # 1=재촬영 필요
y_pred_reshoot = 1 - rule_pass(df)             # 규칙 위반 → 재촬영
cm = confusion_matrix(y_true, y_pred_reshoot, labels=[1,0])
TP,FN,FP,TN = cm.ravel()
f1 = f1_score(y_true, y_pred_reshoot)

# 저장
cm_csv = OUT_EXP / "threshold_confusion_matrix.csv"
pd.DataFrame(cm, index=["True+","True-"], columns=["Pred+","Pred-"]).to_csv(cm_csv, encoding="utf-8-sig")

# 혼동행렬 그래프
plt.figure(figsize=(3.6,3.2))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xticks([0,1], ["Pred + (Reshoot)","Pred -"]); plt.yticks([0,1], ["True +","True -"])
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=11, color="black")
plt.tight_layout()
plt.savefig(OUT_PLOT / "threshold_confusion_matrix.png", dpi=160)
plt.close()

# -----------------------------
# 5) 규칙 위반 지표 디버그(원인 확인)
# -----------------------------
print("\n== Rule failures by video ==")
for _, r in df.iterrows():
    fails = []
    if r["avg_visibility"] < THRESHOLDS["avg_visibility"]: fails.append("avg_visibility")
    if r["visible_ratio"]  < THRESHOLDS["visible_ratio"]:  fails.append("visible_ratio")
    if r["jitter_std"]     > THRESHOLDS["jitter_std"]:     fails.append("jitter_std")
    if r["occlusion_rate"] > THRESHOLDS["occlusion_rate"]: fails.append("occlusion_rate")
    verdict = "FAIL" if len(fails) else "PASS"
    print(f"{r['label']:15s} {verdict}: {', '.join(fails) if fails else '-'}")

# -----------------------------
# 6) 콘솔 요약
# -----------------------------
print("\n[Summary]")
print(f"- Baseline    : {BASELINE_LABEL}")
print(f"- Saved tables: {csv_path}, {lab_csv}, {mrr_csv}, {cm_csv}")
print(f"- Plots folder: {OUT_PLOT}")

print("\n== mRR(mean) ==")
print(mrr[["label","mRR_mean"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

print("\n== TOST(rule-only) ==")
for metric, bound in EQ_BOUNDS.items():
    for _, r in df.iterrows():
        if r["label"] == BASELINE_LABEL: continue
        delta = float(r[metric] - base[metric])
        eq = abs(delta) <= bound
        print(f"{metric:15s} {r['label']:15s} Δ={delta:.4f} eq={eq}")

print("\n== Threshold validity ==")
print(f"Confusion Matrix [[TP,FN],[FP,TN]] = {cm.tolist()}   F1={f1:.3f}")
print("Rules:", THRESHOLDS)
