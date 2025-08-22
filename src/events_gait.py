"""
파일명: src/events_gait.py
 Mediapipe 기반 표준 JSON(time_series)에서 보행 이벤트(HS/TO)를 자동 검출한다.

블록 구성
 0) 라이브러리 임포트: 표준/서드파티 모듈 로드
 1) 신호 준비: toe_y/ankle_y 결측 보간, 평활화
 2) HS 검출: 발끝 y 하강→상승 전환(국소 최소) 근처 추출
 3) TO 검출: 상향 가속 구간(미분/가속 기반 임계)로 추정
 4) 이벤트 정제: 간격/순서 검증, 중복 제거
 5) main/CLI: 입력 JSON/출력 JSON/측(side) 인자 파싱 후 실행

사용 방법
 1) 가상환경 활성화 후 루트에서 실행:
  - python src/events_gait.py --json results/json/sample_gait.json --out results/json/sample_gait_events.json --side right
 2) --side left 로 좌측 계산 가능

입력
 - 표준 JSON (adapters/standard.py 출력): meta + time_series[{time_s, right_toe_y, right_ankle_y, ...}]

출력
 - 입력 JSON에 events 필드를 추가/갱신하여 저장
 - events: {"HS": [t1, t2, ...], "TO": [t1, t2, ...]}

출력 예시
 {
   "meta": {...},
   "time_series": [...],
   "events": {"HS":[0.53,1.07,1.59], "TO":[0.78,1.32,1.83]}
 }
"""
# 0) 라이브러리 임포트 ---------------------------------------------------
import json
import argparse
import numpy as np

# 1) 신호 준비 -----------------------------------------------------------
def _interp_nan(t, y):
    y = np.asarray(y, dtype=float)
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

# 2) HS 검출 --------------------------------------------------------------
def _detect_hs(t, toe_y, vy):
    # 하강(vy<0) → 상승(vy>=0)으로 바뀌는 지점(국소 최저치 근방)
    cross = (vy[:-1] < 0) & (vy[1:] >= 0)
    idx = np.where(np.r_[False, cross])[0]
    # 최소 간격 필터(발 스텝 최소 0.3s 가정)
    times = t[idx]
    filtered = [times[0]] if len(times) else []
    for tt in times[1:]:
        if tt - filtered[-1] >= 0.3:
            filtered.append(tt)
    return filtered

# 3) TO 검출 --------------------------------------------------------------
def _detect_to(t, toe_y, ay):
    # 상승 가속 상위 분위수 임계(기본 80% 분위)
    thr = np.nanpercentile(ay, 80.0)
    idx = np.where(ay >= thr)[0]
    # 과도 검출 방지(0.15s 이내 중복 제거)
    times = t[idx]
    pruned = []
    for tt in times:
        if not pruned or tt - pruned[-1] >= 0.15:
            pruned.append(tt)
    return pruned

# 4) 이벤트 정제 ----------------------------------------------------------
def detect_events_gait(std_json_path, out_path=None, side="right"):
    with open(std_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ts = data["time_series"]
    t  = np.array([x["time_s"] for x in ts], dtype=float)
    toe = np.array([x.get(f"{side}_toe_y", np.nan) for x in ts], dtype=float)

    toe = _interp_nan(t, toe)
    toe = _smooth(toe, win=5)

    vy = np.gradient(toe, t)
    ay = np.gradient(vy, t)

    hs = _detect_hs(t, toe, vy)
    to = _detect_to(t, toe, ay)

    data["events"] = {"HS": [round(x, 3) for x in hs], "TO": [round(x, 3) for x in to]}
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    return data

# 5) main/CLI -------------------------------------------------------------
def _args():
    p = argparse.ArgumentParser(description="보행(가이트) 이벤트 검출")
    p.add_argument("--json", required=True, help="입력 표준 JSON")
    p.add_argument("--out",  default=None, help="출력 JSON(미지정 시 입력 덮어쓰기 안 함)")
    p.add_argument("--side", default="right", choices=["right","left"])
    return p.parse_args()

if __name__ == "__main__":
    a = _args()
    detect_events_gait(a.json, a.out, side=a.side)
