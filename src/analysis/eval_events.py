# -*- coding: utf-8 -*-
"""
파일명: src/analysis/eval_events.py
설명:
  - GT CSV vs Pred CSV를 time_ms 기준 허용오차 창(±tol_ms)으로 1:1 매칭해 평가.
  - 과신전 라벨은 모두 'GR'로 통일.
사용 예:
  python src/analysis/eval_events.py --gt results/gt/gt_hyper.csv --pred results/experiment/pred_hyper.csv --tol-ms 50
"""

import csv, math, argparse
from pathlib import Path
from collections import defaultdict

# ───── 라벨 정규화 ─────
EVENT_ALIAS = {
    "HS": "HS", "HEEL_STRIKE": "HS",
    "TO": "TO", "TOE_OFF": "TO",
    "MS": "MS", "MID_STANCE": "MS",
    "GR": "GR", "GENU_RECURVATUM": "GR",
    "HYPEREXT": "GR", "HYPEREXTENSION": "GR", "HY.EXT.": "GR"
}
def norm_event(e):
    e = str(e).strip().upper().replace("-", "_")
    return EVENT_ALIAS.get(e, e)

def norm_side(s):
    s = str(s).upper().strip()
    return 'L' if s.startswith('L') else ('R' if s.startswith('R') else s)

# 1) CSV 로드 -------------------------------------------------------------------
def load_rows(paths):
    rows=[]
    for p in paths:
        if not Path(p).exists(): continue
        with Path(p).open(newline='', encoding='utf-8') as f:
            r=csv.DictReader(f)
            for x in r:
                y = {k: (v.strip() if isinstance(v, str) else v) for k,v in x.items()}
                y["side"]  = norm_side(y.get("side",""))
                y["event"] = norm_event(y.get("event",""))
                rows.append(y)
    return rows

def group_times(rows):
    g=defaultdict(list)
    for x in rows:
        k=(x["video_id"], x["side"], x["event"])
        t=int(x["time_ms"])
        g[k].append(t)
    for k in g: g[k].sort()
    return g

def group_full(rows):
    g=defaultdict(list)
    for x in rows:
        k=(x["video_id"], x["side"], x["event"])
        t=int(x["time_ms"])
        g[k].append({'t': t, 'row': x})
    for k in g: g[k].sort(key=lambda z: z['t'])
    return g

# 2) 매칭 -----------------------------------------------------------------------
def match(gt_ts, pr_ts, tol_ms):
    """GT와 Pred를 1:1로 가장 가까운 쌍으로 매칭. |Δ| <= tol_ms만 TP."""
    used=[False]*len(pr_ts)
    pairs=[]
    for g in gt_ts:
        best_j, best_d = None, 10**9
        for j,p in enumerate(pr_ts):
            if used[j]: continue
            d=abs(p-g)
            if d<best_d:
                best_d, best_j=d, j
        if best_j is not None and best_d<=tol_ms:
            used[best_j]=True
            pairs.append((g, pr_ts[best_j]))
    TP=len(pairs); FP=used.count(True)-TP; FN=len(gt_ts)-TP
    errs=[abs(b-a) for a,b in pairs]
    return TP,FP,FN,errs,pairs,used

def build_pair_rows(gt_full, pr_full, tol_ms):
    if not gt_full and not pr_full:
        return []
    ref = gt_full[0] if gt_full else pr_full[0]
    side = norm_side(ref['row']['side'])
    event = norm_event(ref['row']['event'])
    gt_full = [x for x in gt_full if norm_side(x['row']['side'])==side and norm_event(x['row']['event'])==event]
    pr_full = [x for x in pr_full if norm_side(x['row']['side'])==side and norm_event(x['row']['event'])==event]
    gt_full = sorted(gt_full, key=lambda x: x['t'])
    pr_full = sorted(pr_full, key=lambda x: x['t'])
    gt = [x['t'] for x in gt_full]; pr = [x['t'] for x in pr_full]
    _,_,_,_,pairs,used = match(gt, pr, tol_ms)

    rows=[]; used_gt=[False]*len(gt_full)
    for g_t, p_t in pairs:
        gi = min((i for i in range(len(gt_full)) if not used_gt[i]), key=lambda i: abs(gt_full[i]['t']-g_t))
        used_gt[gi]=True
        rows.append({
            "video_id": gt_full[gi]['row'].get("video_id",""),
            "side": side, "event": event,
            "gt_time_ms": g_t, "pred_time_ms": p_t,
            "delta_ms": p_t - g_t, "status": "TP"
        })
    for i,x in enumerate(gt_full):
        if not used_gt[i]:
            rows.append({"video_id": x['row'].get("video_id",""),"side": side,"event": event,
                         "gt_time_ms": x['t'], "pred_time_ms": "","delta_ms": "","status": "FN"})
    for j,x in enumerate(pr_full):
        if not used[j]:
            rows.append({"video_id": x['row'].get("video_id",""),"side": side,"event": event,
                         "gt_time_ms": "","pred_time_ms": x['t'],"delta_ms": "","status": "FP"})
    return rows

# 3) 지표 계산 -------------------------------------------------------------------
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp>0 else 0.0
    R=tp/(tp+fn) if tp+fn>0 else 0.0
    F=2*P*R/(P+R) if P+R>0 else 0.0
    return P,R,F
def ms_stats(errs):
    if not errs: return ("-","-","-")
    ms=sorted(errs); n=len(ms)
    med=ms[n//2] if n%2 else (ms[n//2-1]+ms[n//2])/2
    p95=ms[min(n-1, math.ceil(n*0.95)-1)]
    mean=sum(ms)/n
    return (f"{mean:.1f}", f"{med:.1f}", f"{p95:.1f}")

# 4) 메인 ------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt",   nargs="+", required=True, help="GT CSV 파일 경로(들)")
    ap.add_argument("--pred", nargs="+", required=True, help="Pred CSV 파일 경로(들)")
    ap.add_argument("--tol-ms", type=int, default=67, help="허용 오차(ms), 기본=67(ms)≈30fps ±2프레임")
    args = ap.parse_args()

    gt_rows = load_rows(args.gt)
    pr_rows = load_rows(args.pred)

    gt_times = group_times(gt_rows); pr_times = group_times(pr_rows)
    gt_full  = group_full(gt_rows);  pr_full  = group_full(pr_rows)

    by_event=defaultdict(lambda:[0,0,0,[]])
    by_key_rows=[]; pair_rows=[]
    all_keys = sorted(set(gt_times)|set(pr_times))
    for k in all_keys:
        gt_ts = gt_times.get(k, []); pr_ts = pr_times.get(k, [])
        TP,FP,FN,errs,_,_ = match(gt_ts, pr_ts, args.tol_ms)
        P,R,F=prf(TP,FP,FN); m,md,p95=ms_stats(errs)
        vid,side,evt=k
        by_key_rows.append([vid,side,evt,TP,FP,FN,f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])
        e=evt; by_event[e][0]+=TP; by_event[e][1]+=FP; by_event[e][2]+=FN; by_event[e][3]+=errs
        pair_rows.extend(build_pair_rows(gt_full.get(k, []), pr_full.get(k, []), args.tol_ms))

    summary=[["event","TP","FP","FN","P","R","F1","MAE(ms)","Median(ms)","P95(ms)"]]; tot=[0,0,0,[]]
    for e in sorted(by_event):
        tp,fp,fn,errs=by_event[e]; P,R,F=prf(tp,fp,fn); m,md,p95=ms_stats(errs)
        summary.append([e,tp,fp,fn,f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])
        tot[0]+=tp; tot[1]+=fp; tot[2]+=fn; tot[3]+=errs
    P,R,F=prf(tot[0],tot[1],tot[2]); m,md,p95=ms_stats(tot[3])
    summary.append(["ALL",tot[0],tot[1],tot[2],f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])

    outdir=Path("results/experiment"); outdir.mkdir(parents=True, exist_ok=True)
    with (outdir/"events_eval_summary.csv").open("w", newline="", encoding="utf-8") as f: csv.writer(f).writerows(summary)
    with (outdir/"events_eval_bykey.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["video_id","side","event","TP","FP","FN","P","R","F1","MAE(ms)","Median(ms)","P95(ms)"]); w.writerows(by_key_rows)
    pair_rows.sort(key=lambda r: (r["video_id"], r["side"], r["event"], str(r["gt_time_ms"]), str(r["pred_time_ms"])))
    with (outdir/"events_eval_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["video_id","side","event","gt_time_ms","pred_time_ms","delta_ms","status"]); w.writeheader(); w.writerows(pair_rows)
    tp_rows=[r for r in pair_rows if r["status"]=="TP"]
    with (outdir/"events_eval_pairs_tp.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["video_id","side","event","gt_time_ms","pred_time_ms","delta_ms","status"]); w.writeheader(); w.writerows(tp_rows)

    print("[saved] results/experiment/events_eval_summary.csv")
    print("[saved] results/experiment/events_eval_bykey.csv")
    print("[saved] results/experiment/events_eval_pairs.csv")
    print("[saved] results/experiment/events_eval_pairs_tp.csv")

if __name__=="__main__":
    main()
