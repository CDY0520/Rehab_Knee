"""
파일명: src/analysis/eval_events.py

설명:
  - GT 라벨 CSV와 Pred CSV를 비교하여 HS/TO 등 이벤트 검출 성능을 평가한다.
  - 허용 오차(ms) 내에서 GT와 Pred를 1:1 매칭 후 정밀도/재현율/F1, 타이밍 오차를 산출한다.
  - 출력:
      · results/experioment/events_eval_summary.csv
      · results/experioment/events_eval_bykey.csv

사용법 예시:
  python src/analysis/eval_events.py

블록 구성:
  0) import 및 설정(TOL_MS, 경로)
  1) CSV 로드 및 그룹화 함수
  2) GT–Pred 매칭 로직
  3) 성능 지표 계산 (P/R/F1, MAE/Median/P95)
  4) 이벤트별/전체 요약 생성
  5) 결과 CSV 저장 및 출력
  6) CLI 엔트리포인트
"""

import csv, math
from pathlib import Path
from collections import defaultdict

TOL_MS = 67  # 30fps ±2프레임에 해당
EXP_DIR = Path("results/experioment"); EXP_DIR.mkdir(parents=True, exist_ok=True)

def load_rows(paths):
    rows=[]
    for p in paths:
        if not p.exists(): continue
        with p.open(newline='', encoding='utf-8') as f:
            r=csv.DictReader(f)
            for x in r: rows.append({k:v.strip() for k,v in x.items()})
    return rows

def group(rows, is_pred=False):
    g=defaultdict(list)
    for x in rows:
        k=(x["video_id"], x["side"], x["event"])
        t=int(x["time_ms"])
        g[k].append(t)
    for k in g: g[k].sort()
    return g

def match(gt_ts, pr_ts, tol_ms):
    used=[False]*len(pr_ts)
    pairs=[]
    for g in gt_ts:
        best_j, best_d = None, 10**9
        for j,p in enumerate(pr_ts):
            if used[j]: continue
            d=abs(p-g)
            if d<best_d: best_d, best_j=d, j
        if best_j is not None and best_d<=tol_ms:
            used[best_j]=True; pairs.append((g, pr_ts[best_j]))
    TP=len(pairs); FP=used.count(True)-TP; FN=len(gt_ts)-TP
    errs=[abs(b-a) for a,b in pairs]
    return TP,FP,FN,errs

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

def main():
    # 파일 수집
    gt_paths   = list(Path("results/gt").glob("*.csv"))
    pred_paths = list(Path("results/experioment").glob("pred_*.csv"))

    gt  = group(load_rows(gt_paths))
    pr  = group(load_rows(pred_paths))

    by_event=defaultdict(lambda:[0,0,0,[]])  # tp,fp,fn,errs
    by_key_rows=[]
    for k in sorted(set(gt)|set(pr)):
        TP,FP,FN,errs=match(gt.get(k,[]), pr.get(k,[]), TOL_MS)
        P,R,F=prf(TP,FP,FN)
        m,md,p95=ms_stats(errs)
        vid,side,evt=k
        by_key_rows.append([vid,side,evt,TP,FP,FN,f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])
        e=evt; by_event[e][0]+=TP; by_event[e][1]+=FP; by_event[e][2]+=FN; by_event[e][3]+=errs

    # 요약
    summary=[["event","TP","FP","FN","P","R","F1","MAE(ms)","Median(ms)","P95(ms)"]]
    tot=[0,0,0,[]]
    for e in sorted(by_event):
        tp,fp,fn,errs=by_event[e]
        P,R,F=prf(tp,fp,fn); m,md,p95=ms_stats(errs)
        summary.append([e,tp,fp,fn,f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])
        tot[0]+=tp; tot[1]+=fp; tot[2]+=fn; tot[3]+=errs
    P,R,F=prf(tot[0],tot[1],tot[2]); m,md,p95=ms_stats(tot[3])
    summary.append(["ALL",tot[0],tot[1],tot[2],f"{P:.3f}",f"{R:.3f}",f"{F:.3f}",m,md,p95])

    # 저장
    with (EXP_DIR/"events_eval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerows(summary)
    with (EXP_DIR/"events_eval_bykey.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f)
        w.writerow(["video_id","side","event","TP","FP","FN","P","R","F1","MAE(ms)","Median(ms)","P95(ms)"])
        w.writerows(by_key_rows)

    print("[saved] results/experioment/events_eval_summary.csv")
    print("[saved] results/experioment/events_eval_bykey.csv")

if __name__=="__main__":
    main()
