"""
파일명: src/metrics.py
기능: 포즈(프레임별 랜드마크)로부터 무릎·발목 핵심 지표를 계산한다.
     - 공통 각도/벡터 유틸
     - 시계열 추출(관절 각도/발 축 각도/좌표)
     - 무릎 지표: 과신전, 정렬(내반/외반 편차), swing 최대 굴곡, stance:swing 비·좌우차
     - 발목 지표: dorsiflexion/plantarflexion, toe clearance, inversion/eversion, HS 순간 발각도

블록 구성
 0) 라이브러리 임포트
 1) 상수(정상 범위/이상 기준)와 타입
 2) 기하 유틸(벡터·각도·시계열 추출)
 3) 공통 지표(ROM/피크 등)
 4) 무릎 지표 함수
 5) 발목 지표 함수
 6) 리포트 헬퍼(요약 dict 생성)
 7) (옵션) 셀프테스트

사용 방법
 1) 프레임 포맷(JSON 예)
    {
      "fps": 30,
      "frames": [
        {"index":0, "landmarks":{"LEFT_HIP":[x,y,z,v], "LEFT_KNEE":[...], ...}},
        ...
      ]
    }
 2) events.py에서 생성한 stance/swing 마스크, HS/TO 인덱스를 함께 사용하면
    표 기준(stance/swing/TO 근방 등)으로 지표를 안정적으로 계산할 수 있다.

주의
 - 이 코드는 2D 화면 좌표 기반 근사(임상 절대각과 다를 수 있음). 동일 세팅 내 추이 비교에 우선 사용.
 - 도 단위 각도 기준: 무릎/발목의 “중립”을 180°(직선)로 가정한 단순 근사.

참고
 - Mediapipe Pose Landmarks: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
from typing import Dict, List, Tuple, Literal, Optional
import math
import statistics

# 1) 상수(정상 범위/이상 기준)와 타입 --------------------------------------
Joint = Literal[
    "LEFT_HIP","LEFT_KNEE","LEFT_ANKLE","LEFT_HEEL","LEFT_FOOT_INDEX",
    "RIGHT_HIP","RIGHT_KNEE","RIGHT_ANKLE","RIGHT_HEEL","RIGHT_FOOT_INDEX"
]

DEG_NEUTRAL = 180.0  # 관절각 중립(직선) 근사

NORMAL_KNEE = {
    "extension_allowance_deg": 5.0,     # 무릎 신전각: 0°±5° → 180° 기준으로 +5° 초과면 과신전
    "alignment_allowance_deg": 5.0,     # H-K-A 정렬 180°±5°
    "swing_peak_flex_min_deg": 55.0,    # swing 최대 굴곡 정상 하한(55~65°)
    "lr_ratio_diff_pp": 10.0,           # 좌우 stance:swing 차이 허용폭(percentage point)
}

NORMAL_ANKLE = {
    "dorsi_min_deg": 5.0,               # swing phase dorsiflexion ≥ 5~10°
    "plantar_min_deg": 10.0,            # push-off plantarflexion ≥ 10~15° (보수적으로 10°)
    "toe_clearance_min_cm": 1.0,        # 최소 1 cm (권장 1.5~2 cm)
    "invert_evert_norm_deg": (0.0, 5.0),# 정적 0~5° 범위
    "inversion_risk_deg": 10.0,         # >10° 지속 시 위험
}

# 2) 기하 유틸(벡터·각도·시계열 추출) --------------------------------------
def _vec(a, b):
    """a->b 벡터"""
    return (b[0]-a[0], b[1]-a[1], b[2]-a[2])

def _dot(u, v) -> float:
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

def _norm(u) -> float:
    return math.sqrt(_dot(u,u))

def _proj_xy(p):
    """(x,y,z,vis) → 2D(x,y,0)"""
    return (p[0], p[1], 0.0)

def angle_3pt(a, b, c) -> float:
    """
    세 점 A-B-C가 만드는 B의 내각(도).
    예) 무릎각 = HIP-KNEE-ANKLE
    """
    ba = _vec(b, a)
    bc = _vec(b, c)
    denom = (_norm(ba)*_norm(bc)) or 1e-8
    cosv = max(-1.0, min(1.0, _dot(ba, bc)/denom))
    return math.degrees(math.acos(cosv))

def line_angle_xy(p, q) -> float:
    """
    선분 p→q의 화면상 기울기(도). 0°=수평(+x), +는 반시계(위쪽).
    - 발 축(HEEL→FOOT_INDEX) 기울기 등에서 사용.
    """
    dx, dy = (q[0]-p[0]), (q[1]-p[1])
    return math.degrees(math.atan2(dy, dx))

def time_series_angle(frames: List[Dict], a: Joint, b: Joint, c: Joint,
                      vis_thr: float = 0.4) -> List[float]:
    """
    프레임 리스트에서 (a-b-c) 각도 시계열(도) 생성.
    - 일부 프레임에서 랜드마크가 없거나 가시성이 낮으면
      직전 유효각으로 forward-fill. 초기값은 180°(중립)로 시작.
    """
    out: List[float] = []
    last = DEG_NEUTRAL  # 초기값(중립)

    for f in frames:
        lm = f.get("landmarks")
        ok = False
        if lm and (a in lm and b in lm and c in lm):
            A4, B4, C4 = lm[a], lm[b], lm[c]
            # visibility 체크(인덱스 3)
            if (len(A4) >= 4 and len(B4) >= 4 and len(C4) >= 4 and
                A4[3] >= vis_thr and B4[3] >= vis_thr and C4[3] >= vis_thr):
                A = _proj_xy(A4); B = _proj_xy(B4); C = _proj_xy(C4)
                ang = angle_3pt(A, B, C)
                last = ang
                ok = True
        # ok면 방금 계산, 아니면 last 유지
        out.append(last if not ok else last)
    return out

def series_xy(frames: List[Dict], j: Joint) -> Tuple[List[float], List[float]]:
    """특정 관절의 x,y 시계열"""
    xs, ys = [], []
    for f in frames:
        p = f["landmarks"][j]
        xs.append(p[0]); ys.append(p[1])
    return xs, ys

def foot_axis_angle_series(frames: List[Dict], side: Literal["LEFT","RIGHT"]) -> List[float]:
    """
    발 축(HEEL→FOOT_INDEX)의 수평 대비 기울기(도) 시계열.
    - inversion/eversion 근사용 2D 지표(정면 촬영일수록 해석 용이).
    """
    heel = f"{side}_HEEL"; tip = f"{side}_FOOT_INDEX"
    out = []
    for f in frames:
        H = _proj_xy(f["landmarks"][heel])
        T = _proj_xy(f["landmarks"][tip])
        out.append(line_angle_xy(H, T))
    return out

# 3) 공통 지표(ROM/피크 등) ----------------------------------------------
def range_of_motion(series: List[float]) -> float:
    """ROM = max - min"""
    return (max(series) - min(series)) if series else 0.0

def peak_value(series: List[float], mode: Literal["max","min"]="max") -> float:
    """시계열 최대/최소"""
    if not series: return 0.0
    return max(series) if mode=="max" else min(series)

# 4) 무릎 지표 함수 --------------------------------------------------------
def knee_hyperextension_index(knee_deg: List[float],
                              stance_mask: Optional[List[bool]]=None,
                              allowance_deg: float = NORMAL_KNEE["extension_allowance_deg"]) -> Dict[str, float]:
    """
    무릎 과신전 지수:
    - 기준: '중립 180°' 대비 얼마나 더 펴졌는가(180° 초과분).
    - stance 구간만 평가(권장). mask가 없으면 전체에서 평가.
    - 이상 기준: 초과분 > allowance_deg → 과신전 위험.
    반환: {"max_extension_over_deg": x, "is_hyperextension": 0/1}
    """
    if not knee_deg: return {"max_extension_over_deg": 0.0, "is_hyperextension": 0.0}
    seq = [v for i,v in enumerate(knee_deg) if (not stance_mask) or stance_mask[i]]
    if not seq: seq = knee_deg
    max_angle = max(seq)
    over = max(0.0, max_angle - DEG_NEUTRAL)
    return {
        "max_extension_over_deg": over,
        "is_hyperextension": 1.0 if over > allowance_deg else 0.0
    }

def knee_alignment_deviation(knee_frontal_deg: List[float],
                             allowance_deg: float = NORMAL_KNEE["alignment_allowance_deg"]) -> Dict[str, float]:
    """
    무릎 정렬(Hip–Knee–Ankle, 정면) 편차:
    - 180°에서 얼마나 벗어났는지(중앙값 기준).
    - 편차 > allowance → 내반/외반 이상.
    """
    if not knee_frontal_deg: return {"median_dev_deg": 0.0, "is_malalignment": 0.0}
    median_angle = statistics.median(knee_frontal_deg)
    dev = abs(median_angle - DEG_NEUTRAL)
    return {"median_dev_deg": dev, "is_malalignment": 1.0 if dev > allowance_deg else 0.0}

def knee_peak_flexion_swing(knee_deg: List[float],
                            swing_mask: Optional[List[bool]]=None,
                            min_normal_deg: float = NORMAL_KNEE["swing_peak_flex_min_deg"]) -> Dict[str, float]:
    """
    swing phase 최대 굴곡각(작을수록 굴곡):
    - 값을 '중립 180°에서 얼마나 굴곡되었나'로 환산해서 리포트.
    - 굴곡량 flex_amt = 180 - 최소각(=가장 굽힌 상태).
    """
    if not knee_deg: return {"peak_flex_deg": 0.0, "is_reduced": 0.0}
    seq = [v for i,v in enumerate(knee_deg) if (not swing_mask) or swing_mask[i]]
    if not seq: seq = knee_deg
    min_angle = min(seq)
    flex_amt = max(0.0, DEG_NEUTRAL - min_angle)
    return {"peak_flex_deg": flex_amt, "is_reduced": 1.0 if flex_amt < min_normal_deg else 0.0}

def stance_swing_ratio(mask: List[bool]) -> float:
    """단일 다리의 stance:swing 비율(stance/swing). swing=0이면 inf"""
    if not mask: return 0.0
    stance = sum(1 for x in mask if x)
    swing = len(mask) - stance
    return (stance / swing) if swing > 0 else float('inf')

def stance_swing_lr_diff(left_mask: List[bool], right_mask: List[bool],
                         allow_pp: float = NORMAL_KNEE["lr_ratio_diff_pp"]) -> Dict[str, float]:
    """
    좌/우 stance:swing 비율 차이(percentage point):
    - 각 다리 stance 비율(% of cycle) 계산 후 |좌-우| 비교.
    """
    L = sum(1 for x in left_mask if x) / len(left_mask) * 100.0 if left_mask else 0.0
    R = sum(1 for x in right_mask if x) / len(right_mask) * 100.0 if right_mask else 0.0
    diff = abs(L - R)
    return {"left_stance_pct": L, "right_stance_pct": R,
            "abs_diff_pp": diff, "is_asymmetric": 1.0 if diff > allow_pp else 0.0}

# 5) 발목 지표 함수 --------------------------------------------------------
def ankle_angles_series(frames: List[Dict], side: Literal["LEFT","RIGHT"]) -> List[float]:
    """발목(무릎-발목-발끝) 각도 시계열"""
    hip   = f"{side}_KNEE"   # (주의) 발목 각에는 'KNEE-ANKLE-FOOT_INDEX' 사용
    ankle = f"{side}_ANKLE"
    foot  = f"{side}_FOOT_INDEX"
    out = []
    for f in frames:
        H = _proj_xy(f["landmarks"][hip])
        A = _proj_xy(f["landmarks"][ankle])
        F = _proj_xy(f["landmarks"][foot])
        out.append(angle_3pt(H, A, F))
    return out

def ankle_dorsiflexion_max(ankle_deg: List[float],
                           swing_mask: Optional[List[bool]]=None,
                           min_normal_deg: float = NORMAL_ANKLE["dorsi_min_deg"]) -> Dict[str, float]:
    """
    Dorsiflexion 최대(근사):
    - 중립 180°보다 '얼마나 작아졌는가'가 dorsiflexion 양.
      dorsi_amt = 180 - 각도  (양수일수록 dorsiflexion)
    - swing phase에서의 최대값 사용(없으면 전체).
    """
    if not ankle_deg: return {"dorsi_max_deg": 0.0, "is_insufficient": 1.0}
    seq = [v for i,v in enumerate(ankle_deg) if (not swing_mask) or swing_mask[i]]
    if not seq: seq = ankle_deg
    dorsi_max = max(0.0, max(DEG_NEUTRAL - x for x in seq))
    return {"dorsi_max_deg": dorsi_max, "is_insufficient": 1.0 if dorsi_max < min_normal_deg else 0.0}

def ankle_plantarflexion_push_off(ankle_deg: List[float],
                                  stance_mask: Optional[List[bool]]=None,
                                  min_normal_deg: float = NORMAL_ANKLE["plantar_min_deg"]) -> Dict[str, float]:
    """
    Push-off 시 plantarflexion 최대(근사):
    - plantar_amt = 각도 - 180  (양수일수록 plantarflexion)
    - stance phase에서의 최대값 사용(없으면 전체).
    """
    if not ankle_deg: return {"plantar_max_deg": 0.0, "is_insufficient": 1.0}
    seq = [v for i,v in enumerate(ankle_deg) if (not stance_mask) or stance_mask[i]]
    if not seq: seq = ankle_deg
    plantar_max = max(0.0, max(x - DEG_NEUTRAL for x in seq))
    return {"plantar_max_deg": plantar_max, "is_insufficient": 1.0 if plantar_max < min_normal_deg else 0.0}

def toe_clearance_series(foot_index_y: List[float],
                         pelvis_y: Optional[List[float]]=None,
                         scale_cm_per_unit: Optional[float]=None) -> List[float]:
    """
    발끝 상대 높이(근사) 시계열 = 기준선 - foot_y
    - 기준선: pelvis_y 평균(있으면 권장) 또는 foot_y 평균
    - scale_cm_per_unit 제공 시 cm로 환산
    """
    if not foot_index_y: return []
    baseline = statistics.mean(pelvis_y) if pelvis_y else statistics.mean(foot_index_y)
    rel = [baseline - y for y in foot_index_y]
    if scale_cm_per_unit:
        rel = [v*scale_cm_per_unit for v in rel]
    return rel

def toe_clearance_min(foot_index_y: List[float],
                      pelvis_y: Optional[List[float]]=None,
                      scale_cm_per_unit: Optional[float]=None,
                      min_required_cm: float = NORMAL_ANKLE["toe_clearance_min_cm"]) -> Dict[str, float]:
    """
    swing 동안의 최소 toe clearance(근사).
    - 마스크 없이 전체 최솟값을 계산(필요 시 swing 마스크 적용해 필터링 후 이 함수를 호출).
    """
    rel = toe_clearance_series(foot_index_y, pelvis_y, scale_cm_per_unit)
    if not rel: return {"min_clearance": 0.0, "is_low": 1.0}
    mn = min(rel)
    return {"min_clearance": mn, "is_low": 1.0 if (scale_cm_per_unit and mn < min_required_cm) else 0.0}

def inversion_eversion_series(frames: List[Dict], side: Literal["LEFT","RIGHT"]) -> List[float]:
    """
    내번/외번(근사) 시계열:
    - 발 축(HEEL→FOOT_INDEX)의 수평 대비 기울기(도)
    - 정면 촬영일수록 유의미. 값의 부호는 카메라/좌·우에 의존하므로
      보통은 '절대 기울기' 또는 '지속적 편향'을 본다.
    """
    ang = foot_axis_angle_series(frames, side)
    return ang

def inversion_eversion_flags(tilt_deg: List[float],
                             risk_deg: float = NORMAL_ANKLE["inversion_risk_deg"],
                             norm_range = NORMAL_ANKLE["invert_evert_norm_deg"]) -> Dict[str, float]:
    """
    내번/외번 이상 여부:
    - 평균 절대 기울기와 최대 절대 기울기 계산
    - 최대값 > risk_deg 이 오래 지속될 경우 위험(여기선 단순히 최대값 기준)
    """
    if not tilt_deg: return {"mean_abs_deg": 0.0, "max_abs_deg": 0.0, "is_excessive": 0.0}
    abs_vals = [abs(x) for x in tilt_deg]
    mean_abs = statistics.mean(abs_vals)
    mx = max(abs_vals)
    low, high = norm_range
    return {
        "mean_abs_deg": mean_abs,
        "max_abs_deg": mx,
        "is_excessive": 1.0 if mx > risk_deg else 0.0,
        "is_outside_norm_mean": 1.0 if not (low <= mean_abs <= high) else 0.0
    }

def heel_strike_foot_angle(foot_angle_deg_series: List[float],
                           hs_indices: List[int]) -> List[float]:
    """
    Heel strike 순간의 발각도(근사) 배열을 반환.
    - foot_angle = ANKLE-HEEL-FOOT_INDEX 각 등으로 미리 계산한 시계열 사용.
    """
    return [foot_angle_deg_series[i] for i in hs_indices if 0 <= i < len(foot_angle_deg_series)]

# 6) 리포트 헬퍼(요약 dict 생성) -------------------------------------------
def summarize_knee_metrics(frames: List[Dict], side: Literal["LEFT","RIGHT"],
                           stance_mask: Optional[List[bool]]=None,
                           swing_mask: Optional[List[bool]]=None) -> Dict[str, float]:
    """무릎 지표 요약 생성"""
    hip  = f"{side}_HIP"; knee = f"{side}_KNEE"; ank = f"{side}_ANKLE"

    knee_deg = time_series_angle(frames, hip, knee, ank)  # (sagittal 근사)
    # 정렬(정면)은 동일 각도 함수로 계산하되 촬영이 정면일 때 더 신뢰됨
    knee_frontal_deg = knee_deg

    hx = knee_hyperextension_index(knee_deg, stance_mask)
    algn = knee_alignment_deviation(knee_frontal_deg)
    swf = knee_peak_flexion_swing(knee_deg, swing_mask)

    return {
        "knee_rom_deg": range_of_motion(knee_deg),
        "knee_hyperextension_over_deg": hx["max_extension_over_deg"],
        "knee_hyperextension_flag": hx["is_hyperextension"],
        "knee_alignment_dev_deg": algn["median_dev_deg"],
        "knee_alignment_flag": algn["is_malalignment"],
        "knee_swing_peak_flex_deg": swf["peak_flex_deg"],
        "knee_swing_flex_reduced_flag": swf["is_reduced"],
    }

def summarize_ankle_metrics(frames: List[Dict], side: Literal["LEFT","RIGHT"],
                            stance_mask: Optional[List[bool]]=None,
                            swing_mask: Optional[List[bool]]=None,
                            pelvis_y: Optional[List[float]]=None,
                            scale_cm_per_unit: Optional[float]=None) -> Dict[str, float]:
    """발목 지표 요약 생성"""
    ank_deg = ankle_angles_series(frames, side)
    foot_angle = time_series_angle(frames, f"{side}_ANKLE", f"{side}_HEEL", f"{side}_FOOT_INDEX")
    _, toe_y = series_xy(frames, f"{side}_FOOT_INDEX")

    dorsi = ankle_dorsiflexion_max(ank_deg, swing_mask)
    plantar = ankle_plantarflexion_push_off(ank_deg, stance_mask)
    clr = toe_clearance_min(toe_y, pelvis_y, scale_cm_per_unit)
    tilt = inversion_eversion_series(frames, side)
    tilt_flag = inversion_eversion_flags(tilt)

    return {
        "ankle_rom_deg": range_of_motion(ank_deg),
        "dorsi_max_deg": dorsi["dorsi_max_deg"],
        "dorsi_insufficient_flag": dorsi["is_insufficient"],
        "plantar_max_deg": plantar["plantar_max_deg"],
        "plantar_insufficient_flag": plantar["is_insufficient"],
        "toe_clearance_min": clr["min_clearance"],   # scale 주면 cm
        "toe_clearance_low_flag": clr["is_low"] if scale_cm_per_unit else 0.0,
        "inv_evert_mean_abs_deg": tilt_flag["mean_abs_deg"],
        "inv_evert_max_abs_deg": tilt_flag["max_abs_deg"],
        "inv_evert_excessive_flag": tilt_flag["is_excessive"],
        "inv_evert_outside_norm_mean_flag": tilt_flag["is_outside_norm_mean"],
        # 참조용: HS 순간 발각도는 events.py의 HS 인덱스를 입력해 heel_strike_foot_angle()로 별도 호출
    }

# 7) (옵션) 셀프테스트 ------------------------------------------------------
if __name__ == "__main__":
    # 아주 짧은 가짜 프레임 5개로 동작만 확인 (실제 값은 의미 없음)
    frames = []
    for i in range(50):
        # 좌측 랜드마크(단순 파형)
        lh, lk, la = (0.4,0.6,0,1), (0.5,0.70-0.02*math.sin(i/6),0,1), (0.6,0.92-0.02*math.cos(i/7),0,1)
        lh_, lf = (0.58,0.94-0.02*math.sin(i/5),0,1), (0.62,0.98-0.02*math.cos(i/5),0,1)
        frames.append({
            "index": i,
            "landmarks": {
                "LEFT_HIP": lh, "LEFT_KNEE": lk, "LEFT_ANKLE": la,
                "LEFT_HEEL": lh_, "LEFT_FOOT_INDEX": lf
            }
        })

    # 임시 마스크(절반은 stance, 절반은 swing)
    n = len(frames)
    stance_mask = [True if k < n//2 else False for k in range(n)]
    swing_mask  = [not x for x in stance_mask]

    # 무릎/발목 요약
    ksum = summarize_knee_metrics(frames, "LEFT", stance_mask, swing_mask)
    asum = summarize_ankle_metrics(frames, "LEFT", stance_mask, swing_mask, scale_cm_per_unit=None)

    print("[KNEE]", {k: round(v,2) if isinstance(v,float) else v for k,v in ksum.items()})
    print("[ANKLE]", {k: round(v,2) if isinstance(v,float) else v for k,v in asum.items()})
