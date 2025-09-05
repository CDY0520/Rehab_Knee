"""
파일명: src/events.py
 Mediapipe Pose 시계열(.npz)을 입력으로 보행(HS/MS/TO)·STS(Seat-off/Full-stand) 이벤트를 검출하고 지표를 계산한다.
 추가로 무릎/발목 핵심 지표(과신전, stiff-knee, toe-clearance)를 측면 영상 기준으로 산출한다.

블록 구성
 0) 임포트/상수: numpy, pandas(optional), pathlib, json, dataclass
 1) 공통 유틸: npz 로딩, 스무딩, 미분, 내보내기
 2) 보행 이벤트: HS/MS/TO 검출, 보행 지표 계산 + 무릎/발목 지표: 과신전(stance), stiff-knee(swing), toe-clearance(swing)
 3) STS 이벤트: seat-off, full-stand 검출 및 지표
 4) API: detect_gait_events / detect_sts_events / save_events_json / save_events_csv_timeline

사용 예
 python -c "from src.events import detect_gait_events;print(detect_gait_events('results/keypoints/sample_walk.npz','left'))"
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import pandas as pd  # csv 내보내기 선택
except Exception:
    pd = None


# -------------------------------
# 0) 상수(랜드마크 인덱스)
# -------------------------------
L_ANKLE, R_ANKLE = 27, 28
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
NOSE = 0


# -------------------------------
# 파라미터
# -------------------------------
@dataclass
class GaitParams:
    ma_win: int = 5                    # 이동평균 창
    min_step_interval_ms: int = 300    # HS 최소 간격
    hs_prominence: float = 0.003       # 사용 안함(예비)

@dataclass
class STSParams:
    ma_win: int = 7
    min_cycle_ms: int = 1500

@dataclass
class KneeAnkleThresh:
    hyperext_deg: float = 185.0   # 무릎 과신전: 무릎각 > 185°
    stiff_knee_deg: float = 40.0  # swing 최대 굴곡 < 40°
    toe_clear_min: float = 0.012  # 정규화 좌표 기준 최소 toe 상승량(≈1–1.5cm@1.2m)


# -------------------------------
# 1) 공통 유틸
# -------------------------------
def load_npz(path: str | Path):
    d = np.load(path, allow_pickle=True)
    lm_x, lm_y, lm_v = d["lm_x"], d["lm_y"], d["lm_v"]
    t_ms = d["t_ms"]
    valid = d["valid"].astype(bool)
    meta = json.loads(str(d["meta"]))
    return lm_x, lm_y, lm_v, t_ms, valid, meta

def smooth1d(x: np.ndarray, win: int):
    if win <= 1:
        return x
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode="same")

def diff(x: np.ndarray, dt: float):
    dx = np.zeros_like(x, dtype=float)
    if len(x) > 1:
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
    return dx

def _export_json(obj: dict, out: str | Path):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _export_csv(rows: list[dict], out: str | Path):
    if pd is None:
        return
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")


# -------------------------------
# 2) 보행 이벤트 + 무릎/발목 지표
# -------------------------------
def _knee_angle_deg(yx: np.ndarray, hip_i: int, knee_i: int, ankle_i: int) -> np.ndarray:
    # yx: (N, 33, 2)  [y,x]
    a = yx[:, hip_i] - yx[:, knee_i]
    b = yx[:, ankle_i] - yx[:, knee_i]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6
    cos = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def _side_indices(side_hint: str | None):
    if str(side_hint).lower().startswith("l"):
        return L_ANKLE, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE
    if str(side_hint).lower().startswith("r"):
        return R_ANKLE, R_FOOT_INDEX, R_HIP, R_KNEE, R_ANKLE
    return L_ANKLE, L_FOOT_INDEX, L_HIP, L_KNEE, L_ANKLE

def _cycle_segments(hs_idx: list[int], to_idx: list[int]) -> list[tuple[int,int,int]]:
    segs = []
    for i, h in enumerate(hs_idx[:-1]):
        nxt = hs_idx[i + 1]
        tos = [t for t in to_idx if h < t < nxt]
        if not tos:
            continue
        segs.append((h, tos[0], nxt))
    return segs

def _toe_clearance(ly: np.ndarray, toe_i: int, start: int, end: int) -> float:
    seg = ly[start:end + 1, toe_i]
    return float(seg[0] - np.min(seg))  # y는 아래로 커짐 → 시작-최소

def detect_gait_events(npz_path: str, side_hint: str | None = None, params: GaitParams | None = None) -> dict:
    params = params or GaitParams()
    lx, ly, lv, t_ms, valid, meta = load_npz(npz_path)
    dt = np.median(np.diff(t_ms)) / 1000.0 if len(t_ms) > 1 else 1 / max(meta.get("fps", 30), 1)

    ankle_i, toe_i, hip_i, knee_i, _ = _side_indices(side_hint)
    yx = np.stack([ly, lx], axis=-1)

    ank_y = ly[:, ankle_i].copy()
    sig = smooth1d(ank_y, params.ma_win)
    vel = diff(sig, dt)

    # HS: 속도 +→- 전환 근방 로컬 미니마
    sign = np.sign(vel)
    zc = np.where((sign[:-1] > 0) & (sign[1:] < 0))[0] + 1
    hs_idx = []
    last_hs_t = -1e9
    for i in zc:
        s = max(0, i - 2); e = min(len(sig), i + 3)
        if sig[i] <= sig[s:e].min() and (t_ms[i] - last_hs_t) >= params.min_step_interval_ms:
            hs_idx.append(i); last_hs_t = t_ms[i]

    # TO: HS 이후 −→+ 전환점
    to_idx = []
    k = 0
    for i in range(1, len(sign)):
        if sign[i - 1] < 0 and sign[i] > 0:
            while k + 1 < len(hs_idx) and hs_idx[k + 1] < i:
                k += 1
            if k < len(hs_idx) and hs_idx[k] < i:
                to_idx.append(i)

    # MS: HS~TO 사이 무릎각 최대
    knee_deg = _knee_angle_deg(yx, hip_i, knee_i, ankle_i)
    ms_idx = []
    for h in hs_idx:
        t_candidates = [t for t in to_idx if t > h]
        if not t_candidates:
            continue
        t = t_candidates[0]
        if t - h >= 2:
            ms_idx.append(int(np.argmax(knee_deg[h:t + 1]) + h))

    # 보행 지표
    steps = len(hs_idx)
    cadence = (steps / (t_ms[-1] - t_ms[0])) * 60000.0 if steps > 1 else 0.0
    stance_ratios, swing_ratios = [], []
    for i, h in enumerate(hs_idx[:-1]):
        nxt = hs_idx[i + 1]
        tos = [t for t in to_idx if h < t < nxt]
        if not tos:
            continue
        t = tos[0]
        stance = (t_ms[t] - t_ms[h]); swing = (t_ms[nxt] - t_ms[t]); cyc = stance + swing
        if cyc > 0:
            stance_ratios.append(stance / cyc); swing_ratios.append(swing / cyc)

    # 무릎/발목 지표
    ka = KneeAnkleThresh()
    segs = _cycle_segments(hs_idx, to_idx)
    hyperext_flags, stiff_flags, toe_cl_list = [], [], []
    for h, t, nxt in segs:
        hyperext_flags.append(1 if np.any(knee_deg[h:t + 1] > ka.hyperext_deg) else 0)
        swing_max = float(np.max(knee_deg[t:nxt + 1]))
        stiff_flags.append(1 if swing_max < ka.stiff_knee_deg else 0)
        toe_cl_list.append(_toe_clearance(ly, toe_i, t, nxt))

    metrics_knee_ankle = {
        "cycles_eval": len(segs),
        "hyperextension_ratio": round(float(np.mean(hyperext_flags)) if segs else 0.0, 3),
        "stiff_knee_ratio":      round(float(np.mean(stiff_flags)) if segs else 0.0, 3),
        "toe_clear_min_mean":    round(float(np.mean(toe_cl_list)) if segs else 0.0, 4),
        "has_hyperextension":    bool(segs and any(hyperext_flags)),
        "has_stiff_knee":        bool(segs and any(stiff_flags)),
        "low_toe_clearance":     bool(segs and (np.mean(toe_cl_list) < ka.toe_clear_min if segs else False)),
        "thresholds": {
            "hyperext_deg": ka.hyperext_deg,
            "stiff_knee_deg": ka.stiff_knee_deg,
            "toe_clear_min": ka.toe_clear_min,
        },
    }

    result = {
        "task": "gait",
        "npz": str(npz_path),
        "meta": meta,
        "events": {
            "HS_ms": [int(t_ms[i]) for i in hs_idx],
            "TO_ms": [int(t_ms[i]) for i in to_idx],
            "MS_ms": [int(t_ms[i]) for i in ms_idx],
        },
        "metrics": {
            "steps": steps,
            "cadence_spm": round(float(cadence), 2),
            "stance_ratio_mean": round(float(np.mean(stance_ratios)) if stance_ratios else 0.0, 3),
            "swing_ratio_mean": round(float(np.mean(swing_ratios)) if swing_ratios else 0.0, 3),
        },
        "metrics_knee_ankle": metrics_knee_ankle,
    }
    return result


# -------------------------------
# 3) STS 이벤트
# -------------------------------
def detect_sts_events(npz_path: str, params: STSParams | None = None) -> dict:
    params = params or STSParams()
    lx, ly, lv, t_ms, valid, meta = load_npz(npz_path)
    dt = np.median(np.diff(t_ms)) / 1000.0 if len(t_ms) > 1 else 1 / max(meta.get("fps", 30), 1)

    pelvis_y = smooth1d((ly[:, L_HIP] + ly[:, R_HIP]) / 2.0, params.ma_win)
    yx = np.stack([ly, lx], axis=-1)
    knee_deg_l = _knee_angle_deg(yx, L_HIP, L_KNEE, L_ANKLE)
    knee_deg_r = _knee_angle_deg(yx, R_HIP, R_KNEE, R_ANKLE)
    knee_deg = smooth1d((knee_deg_l + knee_deg_r) / 2.0, params.ma_win)

    v = diff(pelvis_y, dt)

    so_idx = []
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] < v[i + 1] and v[i] < -0.02:
            if not so_idx or (t_ms[i] - t_ms[so_idx[-1]]) > params.min_cycle_ms:
                so_idx.append(i)

    fs_idx = []
    kv = diff(knee_deg, dt)
    for s in so_idx:
        for j in range(s + 1, len(v)):
            if abs(v[j]) < 0.005 and (j - s) > int(0.5 / dt) and abs(kv[j]) < 5.0:
                fs_idx.append(j); break

    cycles = min(len(so_idx), len(fs_idx))
    durations = [(t_ms[fs_idx[i]] - t_ms[so_idx[i]]) / 1000.0 for i in range(cycles)]

    result = {
        "task": "sts",
        "npz": str(npz_path),
        "meta": meta,
        "events": {
            "seat_off_ms": [int(t_ms[i]) for i in so_idx],
            "full_stand_ms": [int(t_ms[i]) for i in fs_idx],
        },
        "metrics": {
            "cycles": cycles,
            "mean_cycle_sec": round(float(np.mean(durations)) if durations else 0.0, 3),
        },
    }
    return result


# -------------------------------
# 4) 저장 API
# -------------------------------
def save_events_json(result: dict, out_path: str | Path):
    _export_json(result, out_path)

def save_events_csv_timeline(result: dict, out_csv: str | Path):
    rows = []
    ev = result.get("events", {})
    for k, arr in ev.items():
        for t in arr:
            rows.append({"event": k, "time_ms": int(t)})
    _export_csv(rows, out_csv)
