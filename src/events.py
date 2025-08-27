"""
파일명: src/events.py
기능: 프레임별 포즈 좌표(특히 발/발끝 y, 관절각 시계열)를 이용해
      보행 이벤트(Heel Strike, Toe Off), stance/swing 마스크, 운동 반복(rep)을 검출한다.

블록 구성
 0) 라이브러리 임포트
 1) 시계열 유틸(배열 변환, 미분/속도, 이동평균/평활)
 2) 보행 이벤트 검출: heel strike(HS), toe off(TO), (NEW) hybrid FS fallback
 3) stance/swing 마스크 생성
 4) 운동 이벤트: 관절각 시계열 기반 rep 카운트(peak↔valley)
 5) (옵션) 셀프테스트용 메인

입력
 - fps: 초당 프레임
 - y시계열: HEEL_y, FOOT_INDEX_y 등 (MediaPipe 정규화 y; 화면 아래로 갈수록 값 증가)
 - angle_deg: 관절각 시계열(예: 무릎각; 도 단위)

출력
 - HS/TO 인덱스 리스트
 - stance/swing boolean 마스크
 - (NEW) 하이브리드 FS 이벤트(Event("FS", frame, time_s)) 리스트
 - (rep_count, rep_starts, rep_ends)

참고
 - 보행 이벤트 규칙은 측면 촬영 y축 기준 근사이며, 실제 촬영 세팅에 따라 임계값 튜닝 필요.
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
from typing import List, Tuple
from collections import namedtuple
import numpy as np

# 이벤트 구조체 (type: "FS"/"HS"/"TO" 등, frame, time_s)
Event = namedtuple("Event", ["type", "frame", "time_s"])

# 1) 시계열 유틸 ----------------------------------------------------------
def _series(arr_like) -> np.ndarray:
    """list/tuple 등을 float numpy array로 변환"""
    return np.asarray(arr_like, dtype=float)

def diff(series: List[float]) -> np.ndarray:
    """1차 차분"""
    s = _series(series)
    if s.size == 0:
        return s
    d = np.zeros_like(s)
    d[1:] = s[1:] - s[:-1]
    return d

def velocity(series: List[float], fps: float) -> np.ndarray:
    """수직속도(근사) = 1차 차분 * fps"""
    return diff(series) * float(fps)

def moving_average(series: List[float], k: int=3) -> np.ndarray:
    """간단 이동평균(홀수 권장). k<=1이면 원본 반환."""
    s = _series(series)
    if k <= 1 or s.size == 0:
        return s
    pad = k // 2
    # 양끝 단순 패딩 후 컨볼루션
    sp = np.pad(s, (pad, pad), mode='edge')
    ker = np.ones(k, dtype=float) / k
    return np.convolve(sp, ker, mode='valid')

def local_minima_idx(y: np.ndarray, win: int=2) -> List[int]:
    """
    국소 최소 인덱스 근사: 중심값이 좌/우 win 범위의 최소와 일치할 때.
    (간단 근사이므로 잡음이 크면 moving_average로 먼저 평활 권장)
    """
    out = []
    n = len(y)
    for i in range(n):
        s = max(0, i-win); e = min(n, i+win+1)
        if y[i] == np.min(y[s:e]):
            # 완전히 평평한 구간의 다중 인덱스 방지: 한 구간당 중앙값만 취함
            if not out or i - out[-1] > win:
                out.append(i)
    return out

def local_maxima_idx(y: np.ndarray, win: int=2) -> List[int]:
    """국소 최대 인덱스 근사"""
    out = []
    n = len(y)
    for i in range(n):
        s = max(0, i-win); e = min(n, i+win+1)
        if y[i] == np.max(y[s:e]):
            if not out or i - out[-1] > win:
                out.append(i)
    return out

# 2) 보행 이벤트 검출 ------------------------------------------------------
def detect_heel_strike(heel_y: List[float], fps: float,
                       smooth_k: int=5, min_interval_ms: int=250) -> List[int]:
    """
    Heel Strike(HS) 검출(측면 세팅 가정):
    - 힐 y의 수직속도가 (하강→상승)으로 부호 전환되는 근방에서
      y가 국소 최소(local min)인 프레임을 HS로 가정.
    - smooth_k: 이동평균 커널(잡음 완화), min_interval_ms: 연속 검출 방지 기간.
    """
    y_raw = _series(heel_y)
    y = moving_average(y_raw, k=smooth_k)
    vy = velocity(y, fps)

    minima = set(local_minima_idx(y, win=2))
    hs = []
    refractory = int((min_interval_ms/1000) * fps)

    for i in range(1, len(y)-1):
        # 속도 부호: 음수(내려감) -> 양수(올라감) 전환 지점
        if vy[i-1] < 0 <= vy[i]:
            # 해당 프레임이 local minimum 근처인지 확인
            cand = min(range(max(0,i-2), min(len(y), i+3)), key=lambda k: abs(y[k]-y[i]))
            if cand in minima:
                if not hs or (cand - hs[-1] > refractory):
                    hs.append(cand)
    return hs

def detect_toe_off(foot_index_y: List[float], fps: float,
                   smooth_k: int=5, min_interval_ms: int=250) -> List[int]:
    """
    Toe Off(TO) 검출(측면 세팅 가정):
    - 발끝 y의 수직속도가 (상승→하강)으로 전환되는 근방에서
      y가 국소 최대(local max)인 프레임을 TO로 가정.
    """
    y_raw = _series(foot_index_y)
    y = moving_average(y_raw, k=smooth_k)
    vy = velocity(y, fps)

    maxima = set(local_maxima_idx(y, win=2))
    to = []
    refractory = int((min_interval_ms/1000) * fps)

    for i in range(1, len(y)-1):
        # 속도 부호: 양수(올라감) -> 음수(내려감)
        if vy[i-1] > 0 >= vy[i]:
            cand = min(range(max(0,i-2), min(len(y), i+3)), key=lambda k: abs(y[k]-y[i]))
            if cand in maxima:
                if not to or (cand - to[-1] > refractory):
                    to.append(cand)
    return to

# --- (NEW) Hybrid FS detector (front/side 공통 fallback) ------------------
def detect_steps_hybrid(knee_deg_series: List[float],
                        ankle_y_series: List[float],
                        fps: float,
                        min_gap_ms: int = 300,
                        w_smooth: int = 7) -> List[Event]:
    """
    하이브리드 FS(≈ foot strike) 탐지:
    - 소스1(무릎각): peak → 다음 trough 구간의 trough를 FS 후보로 사용
    - 소스2(발목 y): y의 국소 최소(지면 접촉 근방)를 FS 후보로 사용
    - 두 소스를 병합하고 최소 간격(min_gap_ms)로 중복 제거

    반환: [Event("FS", frame, time_s), ...]
    """
    k = moving_average(knee_deg_series, k=w_smooth)
    a = moving_average(ankle_y_series, k=w_smooth)
    n = len(k)

    # --- 소스1: 무릎각에서 peak/trough 추출 ---
    peaks   = local_maxima_idx(k, win=2)
    troughs = local_minima_idx(k, win=2)

    fs_from_knee = []
    troughs_np = np.array(troughs, dtype=int)
    last = -10**9
    refractory = int((min_gap_ms/1000.0) * fps)

    for p in peaks:
        # peak 이후 가장 가까운 trough 선택
        aft = troughs_np[troughs_np > p]
        if aft.size == 0:
            continue
        t = int(aft[0])
        if t - last >= refractory:
            fs_from_knee.append(t)
            last = t

    # --- 소스2: 발목 y의 국소 최소 ---
    fs_from_ankle = local_minima_idx(a, win=2)

    # --- 병합/정렬/간격 필터 ---
    all_cands = sorted(fs_from_knee + fs_from_ankle)
    merged = []
    last = -10**9
    for f in all_cands:
        if f - last >= refractory:
            merged.append(int(f))
            last = f

    return [Event("FS", f, f/float(fps)) for f in merged]

# 3) stance/swing 마스크 ---------------------------------------------------
def stance_swing_masks(num_frames: int, hs_indices: List[int], to_indices: List[int]) -> Tuple[List[bool], List[bool]]:
    """
    HS→TO 구간을 stance, TO→다음 HS 구간을 swing으로 마킹(한쪽 발 기준).
    - num_frames: 전체 프레임 길이
    """
    stance = [False]*num_frames
    swing  = [False]*num_frames
    hs_sorted = sorted(hs_indices)
    to_sorted = sorted(to_indices)

    # HS와 그 직후 나타나는 TO를 쌍으로 맵핑
    j = 0
    pairs = []
    for h in hs_sorted:
        while j < len(to_sorted) and to_sorted[j] < h:
            j += 1
        if j < len(to_sorted):
            t = to_sorted[j]
            if h < t:
                pairs.append((h, t))
            j += 1

    for (h, t) in pairs:
        # stance: HS~TO
        for k in range(h, min(t, num_frames)):
            stance[k] = True
        # swing: TO~다음 HS
        next_h = next((x for x in hs_sorted if x > t), num_frames)
        for k in range(t, min(next_h, num_frames)):
            swing[k] = True

    return stance, swing

# 4) 운동 이벤트(Rep) ------------------------------------------------------
def count_reps_from_angle(angle_deg: List[float], fps: float,
                          min_rep_sec: float=0.6, prominence: float=5.0,
                          smooth_k: int=3) -> Tuple[int, List[int], List[int]]:
    """
    관절각 시계열에서 반복운동(rep) 횟수 추정.
    - 간단 규칙: valley(최대 굴곡) -> peak(최대 신전) -> 다음 valley 구간을 1 rep으로 간주
    - prominence: peak/valley가 주변평균과 구분되는 최소 차이(도)
    - min_rep_sec: 한 rep 최소 시간(프레임 간격 필터)
    반환: (rep_count, rep_start_indices, rep_end_indices)  # start=end=valley 기준
    """
    a_raw = _series(angle_deg)
    a = moving_average(a_raw, k=smooth_k)
    n = len(a)
    if n == 0:
        return 0, [], []

    # 간이 prominence: 이동평균 대비 차이가 threshold 넘는 국소 extremum만 채택
    base = moving_average(a, k=max(3, smooth_k))
    delta = a - base
    peaks = [i for i in local_maxima_idx(a, win=2) if delta[i] >= prominence]
    valleys = [i for i in local_minima_idx(a, win=2) if -delta[i] >= prominence]

    min_frames = int(min_rep_sec * fps)
    starts, ends = [], []

    vi = 0
    while vi < len(valleys):
        v = valleys[vi]
        # v 이후의 첫 peak
        p = next((idx for idx in peaks if idx > v), None)
        if p is None:
            break
        # 그 peak 이후의 다음 valley
        nv = next((idx for idx in valleys if idx > p), None)
        if nv is None:
            break
        if (nv - v) >= min_frames:
            starts.append(v)
            ends.append(nv)
        vi = valleys.index(nv)  # 다음 valley부터 계속

    return len(starts), starts, ends

# 5) (옵션) 셀프테스트 ------------------------------------------------------
if __name__ == "__main__":
    # 가짜 파형으로 HS/TO/rep/FS 동작을 대략 확인
    fps = 30.0
    t = np.linspace(0, 6, int(6*fps))

    # 발목/힐/발끝 y 파형(사인 + 잡음) — 단지 동작 확인용
    heel_y = 0.5 + 0.1*np.sin(2*np.pi*1.2*t) + 0.01*np.random.randn(t.size)
    ankle_y = 0.5 + 0.1*np.sin(2*np.pi*1.2*t + np.pi/6) + 0.01*np.random.randn(t.size)
    toe_y  = 0.5 + 0.1*np.sin(2*np.pi*1.2*t + np.pi/2) + 0.01*np.random.randn(t.size)

    hs = detect_heel_strike(heel_y.tolist(), fps)
    to = detect_toe_off(toe_y.tolist(), fps)
    stance, swing = stance_swing_masks(len(t), hs, to)

    # 각도 파형(굴곡↔신전) — rep/FS 테스트용
    knee = 160 - 25*np.sin(2*np.pi*0.6*t) + 1.5*np.random.randn(t.size)
    fs_events = detect_steps_hybrid(knee.tolist(), ankle_y.tolist(), fps, min_gap_ms=300, w_smooth=7)

    rc, rs, re = count_reps_from_angle(knee.tolist(), fps, min_rep_sec=0.8, prominence=6)

    print(f"[HS] {hs[:5]} ... (총 {len(hs)})")
    print(f"[TO] {to[:5]} ... (총 {len(to)})")
    print(f"[stance %] {round(100*sum(stance)/len(stance),1)} / [swing %] {round(100*sum(swing)/len(swing),1)}")
    print(f"[FS] {len(fs_events)} events, first 3 = {fs_events[:3]}")
    print(f"[rep] count={rc}, starts={rs[:3]}, ends={re[:3]}")
