"""
파일명: run_qmetrics_compare.py
기능: data/samples의 5개 영상에 대해 qmetrics 계산 → CSV 저장,
     mRR(성능 유지율)·TOST(등가성)·ICC(일치도) 분석 → 표/그래프를 results 하위에 저장.

블록 구성
 0) 설정/입력 탐색
 1) qmetrics 실행 및 원시 결과 저장(results/experiment/qmetrics_compare.csv)
 2) mRR 계산 및 그래프(result/plots)
 3) TOST 등가성 검정(프레임 상세가 있으면 통계, 없으면 규칙판정) 및 그래프
 4) ICC(프레임 상세가 있을 때) 및 그래프
 5) 콘솔 요약
"""

# -----------------------------
# 0) 설정/입력 탐색
# -----------------------------
import os, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(".")
SRC_QMETRICS = PROJECT_ROOT / "src" / "qmetrics.py"
if not SRC_QMETRICS.exists():
    print("src/qmetrics.py 가 필요합니다."); sys.exit(1)
sys.path.insert(0, str(PROJECT_ROOT))
from src.qmetrics import compute_qmetrics

SAMPLE_DIR = PROJECT_ROOT / "data" / "samples"
OUT_EXP = PROJECT_ROOT / "results" / "experiment"
OUT_PLOT = PROJECT_ROOT / "results" / "plots"
OUT_EXP.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

EXPECT = {
    "Baseline-Side":  "base_side.mp4",
    "Baseline-Front": "base_front.mp4",
    "Dark-Clothes":   "dark_clothes.mp4",
    "Low-Light":      "low_light.mp4",
    "Dark+Low-Light": "dark_low.mp4",
}

# 프레임 단위 상세 CSV(선택): result/frames/<label>_frames.csv
DETAIL_DIR = PROJECT_ROOT / "results" / "frames"

HIGH_BETTER = ["avg_visibility", "visible_ratio"]
LOW_BETTER  = ["jitter_std", "occlusion_rate"]
ALL_KEYS = HIGH_BETTER + LOW_BETTER
EQ_BOUNDS = {"avg_visibility":0.05, "visible_ratio":0.05, "jitter_std":0.005, "occlusion_rate":0.15}

# -----------------------------
# 1) qmetrics 실행 및 원시 결과 저장
# -----------------------------
def find_samples():
    vids = {lbl: None for lbl in EXPECT}
    for lbl, fname in EXPECT.items():
        cand = SAMPLE_DIR / fname
        if cand.exists():
            vids[lbl] = str(cand)
        else:
            # 토큰 포함 fallback
            token = Path(fname).stem.lower()
            hits = [p for p in SAMPLE_DIR.glob("*.mp4") if token in p.name.lower()]
            vids[lbl] = str(hits[0]) if hits else None
    missing = [k for k,v in vids.items() if v is None]
    if missing:
        print("누락된 영상:", missing)
    return vids

samples = find_samples()
rows = []
for label, path in samples.items():
    if not path: continue
    m = compute_qmetrics(path, sample_cap=150)
    if "error" in m:
        print(f"[FAIL] {label}: {m['error']}"); continue
    row = {
        "label": label,
        "video": Path(path).name,
        "fps": m.get("fps", np.nan),
        "avg_visibility": m.get("avg_visibility", np.nan),
        "visible_ratio": m.get("visible_ratio", np.nan),
        "bbox_ratio_mean": m.get("bbox_ratio_mean", np.nan),
        "occlusion_rate": m.get("occlusion_rate", np.nan),
        "jitter_std": m.get("jitter_std", np.nan),
    }
    rows.append(row)
    print(f"[OK] {label:15s} | vis={row['avg_visibility']:.3f} vr={row['visible_ratio']:.3f} "
          f"occ={row['occlusion_rate']:.3f} jit={row['jitter_std']:.3f} fps={row['fps']:.1f}")

df = pd.DataFrame(rows).sort_values("label")
csv_path = OUT_EXP / "qmetrics_compare.csv"
df.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"Saved: {csv_path}")

# -----------------------------
# 2) mRR 계산 및 그래프
# -----------------------------
BASELINE_LABEL = "Baseline-Side" if "Baseline-Side" in df["label"].values else df["label"].iloc[0]
base = df[df["label"]==BASELINE_LABEL].iloc[0]

def mrr_val(row, key):
    b, v = float(base[key]), float(row[key])
    if np.isnan(b) or np.isnan(v) or b==0 or v==0: return np.nan
    return (v/b) if key in HIGH_BETTER else (b/v)

mrr = df[["label"]].copy()
for k in ALL_KEYS:
    mrr[f"mRR_{k}"] = df.apply(lambda r: mrr_val(r,k), axis=1)
mrr["mRR_mean"] = mrr[[f"mRR_{k}" for k in ALL_KEYS]].mean(axis=1)
mrr_csv = OUT_EXP / "mrr_table.csv"
mrr.to_csv(mrr_csv, index=False, encoding="utf-8-sig")

plt.figure(figsize=(8,4))
plt.bar(mrr["label"], mrr["mRR_mean"])
plt.axhline(1.0, linestyle="--")
plt.ylabel("mRR mean"); plt.title(f"Relative Robustness vs. {BASELINE_LABEL}")
plt.xticks(rotation=20, ha="right"); plt.tight_layout()
plt.savefig(OUT_PLOT / "mRR_mean.png", dpi=160); plt.close()

# -----------------------------
# 3) TOST 등가성 검정
#    프레임 상세 CSV가 있으면 paired-TOST, 없으면 Δ가 bounds 내인지 규칙판정
# -----------------------------
from scipy.stats import t as tdist

def load_detail(label):
    f = DETAIL_DIR / f"{label}_frames.csv"
    if f.exists():
        d = pd.read_csv(f)
        keep = [c for c in ALL_KEYS if c in d.columns]
        if len(keep)>=2 and len(d)>=10:
            return d[keep].reset_index(drop=True)
    return None

detail_map = {lbl:load_detail(lbl) for lbl in df["label"].values}
def tost_paired(diff, low, high):
    n = len(diff)
    if n<3: return np.nan, np.nan, np.nan, n
    mean = float(np.mean(diff)); sd = float(np.std(diff, ddof=1)); se = sd/np.sqrt(n)
    if se==0: return 0.0,0.0,mean,n
    dfree = n-1
    t1 = (mean - low)/se; p1 = 1 - tdist.cdf(t1, dfree)   # H1: mean>low
    t2 = (mean - high)/se; p2 = tdist.cdf(t2, dfree)      # H1: mean<high
    return float(p1), float(p2), mean, n

tost_rows = []
for metric,bound in EQ_BOUNDS.items():
    for lbl in df["label"].values:
        if lbl==BASELINE_LABEL: continue
        A = detail_map.get(BASELINE_LABEL); B = detail_map.get(lbl)
        mode = "rule-only"; p1=p2=np.nan; n= int(df[df["label"]==lbl].shape[0])
        delta = float(df.loc[df["label"]==lbl, metric].iloc[0] - base[metric])
        if A is not None and B is not None:
            nmin = min(len(A), len(B))
            if nmin>=30:
                diffs = (B[metric].values[:nmin] - A[metric].values[:nmin])
                p1,p2,delta,n = tost_paired(diffs, -bound, +bound); mode="paired-TOST"
        equivalent = (p1<0.05 and p2<0.05) if mode=="paired-TOST" else (abs(delta)<=bound)
        tost_rows.append({"metric":metric,"label":lbl,"ref":BASELINE_LABEL,
                          "delta":round(delta,6),"eq_bound":bound,
                          "p1":p1,"p2":p2,"equivalent":bool(equivalent),"mode":mode,"n_pairs":n})
tost = pd.DataFrame(tost_rows)
tost_csv = OUT_EXP / "tost_table.csv"
tost.to_csv(tost_csv, index=False, encoding="utf-8-sig")

for metric in EQ_BOUNDS.keys():
    sub = tost[tost["metric"]==metric].copy()
    if sub.empty: continue
    colors = ["#2ca02c" if eq else "#d62728" for eq in sub["equivalent"]]
    plt.figure(figsize=(8,3))
    plt.bar(sub["label"], sub["delta"], color=colors)
    b = EQ_BOUNDS[metric]
    plt.axhline(+b, linestyle="--"); plt.axhline(-b, linestyle="--")
    plt.ylabel("delta (cond - base)")
    plt.title(f"TOST delta vs bounds for {metric} (ref={BASELINE_LABEL})")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(OUT_PLOT / f"tost_{metric}.png", dpi=160); plt.close()

# -----------------------------
# 4) ICC(프레임 상세 있을 때)
# -----------------------------
def icc_3_1(matrix):
    X = np.asarray(matrix, float); n,k = X.shape
    if n<3 or k<2: return np.nan
    m_t = X.mean(axis=1, keepdims=True); m_r = X.mean(axis=0, keepdims=True); g = X.mean()
    SSR = ((m_r - g)**2).sum()*n
    SST = ((X - g)**2).sum()
    SSE = ((X - m_t - m_r + g)**2).sum()
    SSTM = SST - SSR
    MSR = SSR/(k-1); MST = SSTM/(n-1); MSE = SSE/((n-1)*(k-1))
    den = MST + (k-1)*MSE
    return float((MST - MSE)/den) if den!=0 else np.nan

icc_rows = []
labels_with_detail = [lbl for lbl,d in detail_map.items() if d is not None]
if len(labels_with_detail)>=2:
    min_len = min(len(detail_map[l]) for l in labels_with_detail)
    if min_len>=30:
        for metric in ALL_KEYS:
            if not all(metric in detail_map[l].columns for l in labels_with_detail): continue
            mat = np.column_stack([detail_map[l][metric].values[:min_len] for l in labels_with_detail])
            val = icc_3_1(mat)
            icc_rows.append({"metric":metric,"n_frames":int(min_len),
                             "n_conditions":len(labels_with_detail),"ICC3_1":round(val,4)})
icc = pd.DataFrame(icc_rows)
icc_csv = OUT_EXP / "icc_table.csv"
icc.to_csv(icc_csv, index=False, encoding="utf-8-sig")

if not icc.empty:
    plt.figure(figsize=(6,3))
    plt.bar(icc["metric"], icc["ICC3_1"])
    plt.axhline(0.90, linestyle="--", label="excellent")
    plt.axhline(0.75, linestyle="--", label="good")
    plt.ylim(0,1.02); plt.legend()
    plt.title(f"ICC(3,1) across conditions (n={int(icc['n_frames'].iloc[0])} frames)")
    plt.tight_layout()
    plt.savefig(OUT_PLOT / "icc_bar.png", dpi=160); plt.close()

# -----------------------------
# 5) 콘솔 요약 (mRR · TOST · ICC 한 번에)
# -----------------------------
def _fmt(x, nd=3):
    return "nan" if (x is None or (isinstance(x,float) and np.isnan(x))) else f"{x:.{nd}f}"

print("\n[Summary]")
print(f"- Baseline : {BASELINE_LABEL}")
print(f"- Tables   : {csv_path}, {mrr_csv}, {tost_csv}, {icc_csv}")
print(f"- Plots    : {OUT_PLOT}")

# 5-1) mRR 요약
mrr_view = mrr[["label","mRR_mean"]].copy()
mrr_view["mRR_mean"] = mrr_view["mRR_mean"].apply(lambda v: float(v))
print("\n== mRR (mean Relative Robustness) ==")
print(mrr_view.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# 5-2) TOST 요약: 라벨별 등가성 통과 비율과 실패 지표
if not tost.empty:
    g = tost.groupby("label", as_index=False).agg(
        eq_rate=("equivalent", lambda s: np.mean(s.astype(int))),
        fails=("equivalent", lambda s: int((~s).sum()))
    )
    # 실패 지표 모음
    fail_map = (tost[~tost["equivalent"]]
                .groupby("label")["metric"]
                .apply(lambda s: ",".join(sorted(s.unique())))
                .to_dict())
    g["failed_metrics"] = g["label"].map(fail_map).fillna("")
    # 소수점 포맷
    g["eq_rate"] = g["eq_rate"].apply(lambda v: float(v))
    print("\n== TOST equivalence (per label) ==")
    print(g[["label","eq_rate","fails","failed_metrics"]]
          .to_string(index=False,
                     formatters={"eq_rate":lambda x: f"{x:.2f}"}))
else:
    print("\n== TOST equivalence ==")
    print("no TOST rows (상세 프레임 CSV 없음 또는 계산 불가)")

# 5-3) ICC 요약
if not icc.empty:
    print("\n== ICC(3,1) across conditions ==")
    print(icc.to_string(index=False))
    # 간단 결론
    ok = (icc["ICC3_1"] >= 0.90).sum()
    good = ((icc["ICC3_1"] >= 0.75) & (icc["ICC3_1"] < 0.90)).sum()
    print(f"\nICC status: excellent ≥0.90 → {ok} metrics, good 0.75–0.90 → {good} metrics")
else:
    print("\n== ICC(3,1) ==")
    print("no ICC rows (상세 프레임 CSV 부족)")

print("\n[Decision hints]")
print(" - mRR_mean ≥ 0.95 → 성능 유지 양호")
print(" - TOST eq_rate ≥ 0.75 → 등가성 양호, failed_metrics 확인")
print(" - ICC ≥ 0.90 → 조건 간 일치 우수")
