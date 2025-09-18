"""
파일명: src/analysis/eval_events.py
설명:
  - GT CSV vs Pred CSV를 time_ms 기준 허용오차 창으로 1:1 매칭해 평가한다.
  - 출력:
      · results/experiment/events_eval_summary.csv
      · results/experiment/events_eval_bykey.csv
      · results/experiment/events_eval_pairs.csv  ← GT·Pred를 한 행에 정렬해 비교
블록 구성:
  0) import 및 설정(TOL_MS, 경로)
  1) CSV 로드 유틸(전체 행 / 그룹화)
  2) 매칭 로직(요약용 + 페어 테이블용)
  3) 지표 계산(P/R/F1, MAE/Median/P95)
  4) 요약/바이키/페어 CSV 저장
  5) CLI 엔트리포인트
"""

import csv, math
from pathlib import Path
from collections import defaultdict

TOL_MS = 67  # 30fps ±2프레임
EXP_DIR = Path("results/experiment"); EXP_DIR.mkdir(parents=True, exist_ok=True)

# 1) CSV 로드 -------------------------------------------------------------------
def load_rows(paths):
    rows=[]
    for p in paths:
        if not p.exists(): continue
        with p.open(newline='', encoding='utf-8') as f:
            r=csv.DictReader(f)
            for x in r:
                rows.append({k:v.strip() for k,v in x.items()})
    return rows

def group_times(rows):
    """(vid,side,event) -> 정렬된 time_ms 배열"""
    g=defaultdict(list)
    for x in rows:
        k=(x["video_id"], x["side"], x["event"])
        t=int(x["time_ms"])
        g[k].append(t)
    for k in g: g[k].sort()
    return g

def group_full(rows):
    """(vid,side,event) -> [{'t':time_ms, 'row':orig_row}, ...] (t 오름차순)"""
    g=defaultdict(list)
    for x in rows:
        k=(x["video_id"], x["side"], x["event"])
        t=int(x["time_ms"])
        g[k].append({'t': t, 'row': x})
    for k in g: g[k].sort(key=lambda z: z['t'])
    return g

def norm_side(s):
    s = str(s).upper().strip()
    return 'L' if s.startswith('L') else ('R' if s.startswith('R') else None)

# 2) 매칭 -----------------------------------------------------------------------
def match(gt_ts, pr_ts, tol_ms):
    used=[False]*len(pr_ts)
    pairs=[]
    for g in gt_ts:
        best_j, best_d = None, 10**9
        for j,p in enumerate(pr_ts):
            if used[j]:
                continue
            d=abs(p-g)
            if d<best_d:
                best_d, best_j=d, j
        if best_j is not None and best_d<=tol_ms:
            used[best_j]=True
            pairs.append((g, pr_ts[best_j]))
    TP=len(pairs); FP=used.count(True)-TP; FN=len(gt_ts)-TP
    errs=[abs(b-a) for a,b in pairs]
    return TP,FP,FN,errs,pairs,used

def build_pair_rows(gt_full, pr_full, tol_ms, start_ms=None, end_ms=None):
    """
    gt_full/pr_full : group_full()에서 특정 (side,event) 묶음으로 넘어오는 리스트라고 가정.
    여기서도 방어적으로 side/event/시간범위 필터를 한 번 더 적용.
    """
    if not gt_full and not pr_full:
        return []

    # 1) 기준 side/event 결정
    ref = gt_full[0] if gt_full else pr_full[0]
    side = norm_side(ref['row']['side'])
    event = ref['row']['event']

    # 2) side/event 정규화·필터
    gt_full = [x for x in gt_full if norm_side(x['row']['side'])==side and x['row']['event']==event]
    pr_full = [x for x in pr_full if norm_side(x['row']['side'])==side and x['row']['event']==event]

    # 3) 시간 윈도우 필터(선택)
    if start_ms is not None:
        gt_full = [x for x in gt_full if x['t']>=start_ms]
        pr_full = [x for x in pr_full if x['t']>=start_ms]
    if end_ms is not None:
        gt_full = [x for x in gt_full if x['t']<=end_ms]
        pr_full = [x for x in pr_full if x['t']<=end_ms]

    # 4) 시간 정렬 후 매칭
    gt_full = sorted(gt_full, key=lambda x: x['t'])
    pr_full = sorted(pr_full, key=lambda x: x['t'])
    gt = [x['t'] for x in gt_full]
    pr = [x['t'] for x in pr_full]

    _,_,_,_,pairs,used = match(gt, pr, tol_ms)

    rows=[]
    used_gt=[False]*len(gt_full)

    # TP
    for g_t, p_t in pairs:
        gi = min((i for i in range(len(gt_full)) if not used_gt[i]), key=lambda i: abs(gt_full[i]['t']-g_t))
        # pred 쪽은 p_t와 가장 가까운 미사용 인덱스 선택
        pj = min((j for j in range(len(pr_full)) if used[j]), key=lambda j: abs(pr_full[j]['t']-p_t))
        used_gt[gi]=True
        rows.append({
            "video_id": gt_full[gi]['row']["video_id"] if "video_id" in gt_full[gi]['row'] else "",
            "side":     side,
            "event":    event,
            "gt_time_ms":  g_t,
            "pred_time_ms": p_t,
            "delta_ms":    p_t - g_t,
            "status": "TP"
        })

    # FN
    for i,x in enumerate(gt_full):
        if not used_gt[i]:
            rows.append({
                "video_id": x['row'].get("video_id",""),
                "side":     side,
                "event":    event,
                "gt_time_ms":  x['t'],
                "pred_time_ms": "",
                "delta_ms":    "",
                "status": "FN"
            })

    # FP
    for j,x in enumerate(pr_full):
        if not used[j]:
            rows.append({
                "video_id": x['row'].get("video_id",""),
                "side":     side,
                "event":    event,
                "gt_time_ms":  "",
                "pred_time_ms": x['t'],
                "delta_ms":    "",
                "status": "FP"
            })
    return rows


# 3) 지표 계산 -------------------------------------------------------------------
def prf(tp,fp,fn):
    P=tp/(tp+fp) if tp+fp>0 else 0.0
    R=tp/(tp+fn) if tp+fn>0 else 0.0
    F=2*P*R/(P+R) if P+R>0 else 0.0
    return P,R,F

def ms_stats(errs):
    if not errs: return ("-","-","-")
    ms=sorted(errs)
    n=len(ms)
    med=ms[n//2] if n%2 else (ms[n//2-1]+ms[n//2])/2
    p95=ms[min(n-1, math.ceil(n*0.95)-1)]
    mean=sum(ms)/n
    return (f"{mean:.1f}", f"{med:.1f}", f"{p95:.1f}")

# 4) 메인 ------------------------------------------------------------------------
def main():
    # 파일 수집
    gt_paths   = list(Path("results/gt").glob("*.csv"))
    pred_paths = list(Path("results/experiment").glob("pred_*.csv"))

    gt_rows = load_rows(gt_paths)
    pr_rows = load_rows(pred_paths)

    gt_times = group_times(gt_rows)
    pr_times = group_times(pr_rows)

    gt_full  = group_full(gt_rows)
    pr_full  = group_full(pr_rows)

    # 4-1) by-key 및 summary
    by_event=defaultdict(lambda:[0,0,0,[]])  # tp,fp,fn,errs
    by_key_rows=[]
    pair_rows=[]

    all_keys = sorted(set(gt_times)|set(pr_times))
    for k in all_keys:
        gt_ts = gt_times.get(k, [])
        pr_ts = pr_times.get(k, [])
        TP,FP,FN,errs,_,_ = match(gt_ts, pr_ts, TOL_MS)

        P,R,F=prf(TP,FP,FN)
        m,md,p95=ms_stats(errs)
        vid,side,evt=k
        by_key_rows.append([vid,side,evt,TP,FP,FN,f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])

        e=evt; by_event[e][0]+=TP; by_event[e][1]+=FP; by_event[e][2]+=FN; by_event[e][3]+=errs

        # 4-2) 페어 테이블 행 추가
        pair_rows.extend(build_pair_rows(gt_full.get(k, []), pr_full.get(k, []), TOL_MS))

    # 요약 저장
    summary=[["event","TP","FP","FN","P","R","F1","MAE(ms)","Median(ms)","P95(ms)"]]
    tot=[0,0,0,[]]
    for e in sorted(by_event):
        tp,fp,fn,errs=by_event[e]
        P,R,F=prf(tp,fp,fn); m,md,p95=ms_stats(errs)
        summary.append([e,tp,fp,fn,f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])
        tot[0]+=tp; tot[1]+=fp; tot[2]+=fn; tot[3]+=errs
    P,R,F=prf(tot[0],tot[1],tot[2]); m,md,p95=ms_stats(tot[3])
    summary.append(["ALL",tot[0],tot[1],tot[2],f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])

    with (EXP_DIR/"events_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerows(summary)
    with (EXP_DIR/"events_eval_bykey.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["video_id","side","event","TP","FP","FN","P","R","F1","MAE(ms)","Median(ms)","P95(ms)"])
        w.writerows(by_key_rows)

    # 페어 테이블 저장 (핵심)
    pair_rows.sort(key=lambda r: (r["video_id"], r["side"], r["event"], str(r["gt_time_ms"]), str(r["pred_time_ms"])))
    with (EXP_DIR/"events_eval_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=[
            "video_id","side","event","gt_time_ms","pred_time_ms","delta_ms","status"
        ])
        w.writeheader(); w.writerows(pair_rows)

    # ⬇️ GT와 Pred가 실제 매칭된 경우(TP)만 저장
    tp_rows = [r for r in pair_rows if r["status"] == "TP"]
    with (EXP_DIR / "events_eval_pairs_tp.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "video_id", "side", "event", "gt_time_ms", "pred_time_ms", "delta_ms", "status"
        ])
        w.writeheader();
        w.writerows(tp_rows)

    print("[saved] results/experiment/events_eval_summary.csv")
    print("[saved] results/experiment/events_eval_bykey.csv")
    print("[saved] results/experiment/events_eval_pairs.csv")
    print("[saved] results/experiment/events_eval_pairs_tp.csv")

if __name__=="__main__":
    main()
