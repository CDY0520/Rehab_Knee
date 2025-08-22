"""
파일명: src/events_exercise.py
 Mediapipe 기반 표준 JSON에서 운동 수행 영상의 이벤트(반복/피크/저점)를 검출한다.
 (예: 무릎 굴곡/신전 운동, 스쿼트, 레그 익스텐션 등 단일 관절 또는 간단 복합관절 운동)

블록 구성
 0) 라이브러리 임포트
 1) 타깃 신호 선택: 기본은 knee_angle (side별)
 2) 전처리: 결측 보간, 평활화, 정규화
 3) 반복(rep) 검출: 임계(상/하한) 교차 + 히스테리시스 → up/down 구간 카운트
 4) 피크/저점 기록: 각 rep의 최대/최소 각도, ROM
 5) main/CLI

사용 방법
  - python src/events_exercise.py --json results/json/sample_exercise.json --out results/json/sample_exercise_events.json --side right --low 40 --high 70
  (각도 기준 저/고 임계치(도) 지정. 없으면 데이터 기반 자동 추정)

입력
 - 표준 JSON: time_series[{time_s, right_knee_angle, ...}]

출력
 - events: {
     "reps": 10,
     "segments": [
       {"start":0.50,"peak":0.83,"end":1.14,"peak_angle":82.3,"min_angle":45.2,"rom":37.1},
       ...
     ]
   }

출력 예시
 {
   "meta": {...,"task":"exercise"},
   "time_series": [...],
   "events": {"reps": 3,
              "segments": [{"start":0.52,"peak":0.90,"end":1.34,"peak_angle":88.4,"min_angle":41.9,"rom":46.5}]}
 }
"""
# 0) 라이브러리 임포트 ---------------------------------------------------
import json
import argparse
import numpy as np

# 1) 타깃 신호 선택 -------------------------------------------------------
def _extract_angle(ts, side="right", key="knee"):
    col = f"{side}_{key}_angle"
    arr = np.array([x.get(col, np.nan) for x in ts], dtype=float)
    return arr

def _interp_nan(t, y):
    m = np.isnan(y)
    if m.any():
        y[m] = np.interp(t[m], t[~m], y[~m])
    return y

def _smooth(y, win=5):
    if win <= 1:
        return y
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(win) / win
    return np.convolve(ypad, ker, mode="valid")

# 2) 반복 검출(히스테리시스) ----------------------------------------------
def _auto_thresholds(a):
    # 데이터 분포 기반 간단 추정: 하위 20%, 상위 80% 분위
    lo = float(np.nanpercentile(a, 20))
    hi = float(np.nanpercentile(a, 80))
    # 최소 간격 보장
    if hi - lo < 5:
        hi = lo + 5
    return lo, hi

def _segment_reps(t, a, low=None, high=None, min_dur=0.3):
    if low is None or high is None:
        low, high = _auto_thresholds(a)
    state = "idle"
    segs = []
    start = peak_idx = None

    for i in range(len(t)):
        val = a[i]
        if state == "idle":
            if val >= high:
                state = "up"
                start = i
                peak_idx = i
        elif state == "up":
            if val > a[peak_idx]:
                peak_idx = i
            if val <= low:
                # 한 사이클 종료
                dur = t[i] - t[start]
                if dur >= min_dur:
                    segs.append({
                        "start": round(t[start],3),
                        "peak":  round(t[peak_idx],3),
                        "end":   round(t[i],3),
                        "peak_angle": float(a[peak_idx]),
                        "min_angle":  float(np.nanmin(a[start:i+1])),
                        "rom": float(a[peak_idx] - np.nanmin(a[start:i+1])),
                    })
                state = "idle"
                start = peak_idx = None
    return segs, low, high

# 3) 엔트리 포인트 --------------------------------------------------------
def detect_events_exercise(std_json_path, out_path=None, side="right", joint="knee", low=None, high=None):
    with open(std_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ts = data["time_series"]
    t  = np.array([x["time_s"] for x in ts], dtype=float)
    a  = _extract_angle(ts, side=side, key=joint)

    a = _interp_nan(t, a)
    a = _smooth(a, win=5)

    segs, lo, hi = _segment_reps(t, a, low=low, high=high, min_dur=0.3)
    data["events"] = {"reps": len(segs), "segments": segs, "thresholds": {"low": lo, "high": hi}}

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# 4) main/CLI -------------------------------------------------------------
def _args():
    p = argparse.ArgumentParser(description="운동 수행 이벤트(반복/피크/저점) 검출")
    p.add_argument("--json", required=True, help="입력 표준 JSON")
    p.add_argument("--out",  default=None, help="출력 JSON")
    p.add_argument("--side", default="right", choices=["right","left"])
    p.add_argument("--joint", default="knee", choices=["knee","hip","ankle"])
    p.add_argument("--low",  type=float, default=None, help="하한 임계(도)")
    p.add_argument("--high", type=float, default=None, help="상한 임계(도)")
    return p.parse_args()

if __name__ == "__main__":
    a = _args()
    detect_events_exercise(a.json, a.out, side=a.side, joint=a.joint, low=a.low, high=a.high)
