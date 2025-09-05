"""
파일명: src/events_sts.py
 Mediapipe 기반 표준 JSON에서 STS(sit-to-stand) 이벤트를 검출한다.
 (앉은 자세 → 일어섬 → 다시 앉음까지 1사이클 / 반복 횟수 산출)

블록 구성
 0) 라이브러리 임포트
 1) 신호 선택: 엉덩이(hip) 각도 또는 Y좌표를 STS 지표로 사용
 2) 평활화/정규화: 개체/카메라 차이 보정
 3) 이벤트 검출: 저점(앉음)→상승→고점(서있음)→하강 패턴 인식
 4) 사이클/반복 산출: 시작/최고/종료 타임스탬프 추출
 5) main/CLI

사용 방법
  - python src/events_sts.py --json results/json/sample_sts.json --out results/json/sample_sts_events.json --side right

입력
 - 표준 JSON: time_series[{time_s, right_hip_angle 또는 right_hip_y 등}]

출력
 - events: {
     "STS": [
       {"start": 0.53, "peak": 1.02, "end": 1.50},
       ...
     ],
     "reps": 3
   }

출력 예시
 {
   "meta": {...,"task":"sts"},
   "time_series": [...],
   "events": {
     "STS": [{"start":0.62,"peak":1.10,"end":1.54}],
     "reps": 1
   }
 }
"""
# 0) 라이브러리 임포트 ---------------------------------------------------
import json
import argparse
import numpy as np

# 1) 신호 선택 ------------------------------------------------------------
def _choose_signal(ts, side="right"):
    # 우선순위: hip_angle → ankle_y (대체) → toe_y (보조)
    hip = np.array([x.get(f"{side}_hip_angle", np.nan) for x in ts], dtype=float)
    if np.isfinite(hip).sum() > 10:
        return hip, "angle"  # 각도 기반
    ank = np.array([x.get(f"{side}_ankle_y", np.nan) for x in ts], dtype=float)
    if np.isfinite(ank).sum() > 10:
        return ank, "y"      # 좌표 기반
    toe = np.array([x.get(f"{side}_toe_y", np.nan) for x in ts], dtype=float)
    return toe, "y"

def _interp_nan(t, y):
    m = np.isnan(y)
    if m.any():
        y[m] = np.interp(t[m], t[~m], y[~m])
    return y

def _smooth(y, win=7):
    if win <= 1:
        return y
    pad = win // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(win) / win
    return np.convolve(ypad, ker, mode="valid")

# 2) 패턴 검출 ------------------------------------------------------------
def _detect_sts_cycles(t, s, mode="angle"):
    """
    단순 규칙:
     - angle 모드: '앉음(각도 큰 상태)' → '서있음(각도 작은 상태)' 전환
     - y 모드: '앉음(y 큰 상태)' → '서있음(y 작은 상태)' 전환
    1차 미분부호 전환과 극값으로 start/peak/end 찾기
    """
    # 정규화
    s = (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-9)
    ds = np.gradient(s, t)
    # 극값 인근 인덱스
    minima = (np.r_[True, (ds[1:] >= 0) & (ds[:-1] < 0)] & np.r_[ (ds[:-1] > 0) & (ds[1:] <= 0), True]).nonzero()[0]
    maxima = (np.r_[True, (ds[1:] <= 0) & (ds[:-1] > 0)] & np.r_[ (ds[:-1] < 0) & (ds[1:] >= 0), True]).nonzero()[0]

    # 간단한 매칭(저점→고점→저점)을 사이클로 정의
    mins = sorted(minima.tolist())
    maxs = sorted(maxima.tolist())
    cycles = []
    i = j = 0
    while i < len(mins)-1 and j < len(maxs):
        start_idx = mins[i]
        # start 이후 나타나는 최대점
        cand_max = [mx for mx in maxs if mx > start_idx]
        if not cand_max:
            break
        peak_idx = cand_max[0]
        # peak 이후 다음 최소점
        cand_min2 = [mn for mn in mins if mn > peak_idx]
        if not cand_min2:
            break
        end_idx = cand_min2[0]
        # 최소 길이/진폭 조건
        if t[end_idx]-t[start_idx] >= 0.4 and (s[peak_idx]-s[start_idx]) >= 0.05:
            cycles.append({"start": round(t[start_idx],3),
                           "peak":  round(t[peak_idx],3),
                           "end":   round(t[end_idx],3)})
        i = mins.index(end_idx)  # 다음 사이클 탐색
        j = max(j, maxs.index(peak_idx)+1)
    return cycles

# 3) 엔트리 포인트 --------------------------------------------------------
def detect_events_sts(std_json_path, out_path=None, side="right"):
    with open(std_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ts = data["time_series"]
    t  = np.array([x["time_s"] for x in ts], dtype=float)

    sig, mode = _choose_signal(ts, side)
    sig = _interp_nan(t, sig)
    sig = _smooth(sig, win=7)

    cycles = _detect_sts_cycles(t, sig, mode=mode)
    data["events"] = {"STS": cycles, "reps": len(cycles)}

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# 4) main/CLI -------------------------------------------------------------
def _args():
    p = argparse.ArgumentParser(description="STS 이벤트/반복 검출")
    p.add_argument("--json", required=True, help="입력 표준 JSON")
    p.add_argument("--out",  default=None, help="출력 JSON")
    p.add_argument("--side", default="right", choices=["right","left"])
    return p.parse_args()

if __name__ == "__main__":
    a = _args()
    detect_events_sts(a.json, a.out, side=a.side)
