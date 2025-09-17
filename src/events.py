"""
파일명: src/events.py
설명(업데이트, 전체본):
  - Mediapipe Pose 시계열(.npz)에서 보행(Gait)·STS 이벤트/지표 계산.
  - HS/TO/MS 전부 새 규칙으로 교체:
      · diff_ht(t)=heel_y−toe_y 파형으로 각 사이드의 주기 내 이벤트를
        MS(플랫 구간 중앙) → TO(diff_ht 음의 피크) → HS(diff_ht 양의 피크) 순서로 검출.
      · diff_ht 진폭이 전 구간에서 극히 작으면(마비·플랫풋) 해당 사이드 이벤트 0건 처리.
  - 출력: JSON 요약, 타임라인 CSV, 지표 CSV, 스텝/스트라이드 CSV (results/reports)
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

# ───────────────────────────────
# Mediapipe landmark indices
# ───────────────────────────────
L_ANKLE, R_ANKLE = 27, 28
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_HEEL, R_HEEL = 29, 30  # 뒤꿈치

# ───────────────────────────────
# Parameters
# ───────────────────────────────
@dataclass
class GaitParams:
    # 품질 가드
    ma_win: int = 5
    min_step_interval_ms: int = 200
    vis_min: float = 0.6
    bbox_min: float = 0.01
    bbox_max: float = 0.6

    # diff_ht 기반 규칙
    ms_zero_eps_rel: float = 0.003      # MS 후보: |heel−toe| ≤ eps
    prom_rel: float = 0.006             # TO/HS 피크 prominence 최소
    min_gap_ms: int = 200               # 이벤트 간 최소 간격
    min_diffht_amp_rel: float = 0.004   # 전체 진폭이 이보다 작으면 이벤트 0건 처리

    # 보조 지표/지표 산출
    near_ext_flex_deg: float = 8.0
    stiff_knee_deg: float = 40.0
    toe_clear_min: float = 0.010

    # GENU_RECURVATUM(과신전 스냅) 탐지
    z_omega_thresh: float = 1.3
    z_jerk_thresh: float = 1.5
    stance_mid_lo: float = 0.18
    stance_mid_hi: float = 0.55
    refractory_frac: float = 0.15

@dataclass
class STSParams:
    ma_win: int = 7
    min_cycle_ms: int = 1500
    v_pelvis_drop: float = -0.02
    v_pelvis_stop: float = 0.005
    kv_knee_stop: float = 5.0  # °/s

# ───────────────────────────────
# IO / math utils
# ───────────────────────────────
def load_npz(path: str | Path):
    d = np.load(path, allow_pickle=True)
    meta_raw = d["meta"]
    meta = json.loads(str(meta_raw.item() if hasattr(meta_raw, "item") else meta_raw))
    def opt(name): return d[name] if name in d.files else None
    return {
        "lm_x": d["lm_x"], "lm_y": d["lm_y"], "lm_v": d["lm_v"],
        "t_ms": d["t_ms"], "valid": d["valid"].astype(bool),
        "vis_mean": opt("vis_mean"), "n_visible": opt("n_visible"),
        "bbox_ratio": opt("bbox_ratio"), "jitter": opt("jitter"),
        "quality_ok": opt("quality_ok"), "meta": meta,
    }

def smooth1d(x: np.ndarray, win: int):
    if win <= 1: return x
    k = np.ones(win, dtype=float) / win
    return np.convolve(x, k, mode="same")

def diff(x: np.ndarray, dt: float):
    dx = np.zeros_like(x, dtype=float)
    if len(x) > 1:
        dx[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
        dx[0] = (x[1] - x[0]) / dt
        dx[-1] = (x[-1] - x[-2]) / dt
    return dx

def vel(sig: np.ndarray, fps: float): return np.gradient(sig) * fps

def _export_json(obj: dict, out: str | Path):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _export_csv(rows: list[dict], out: str | Path):
    if pd is None: return
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

def _robust_z(x: np.ndarray, eps: float = 1e-9):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + eps
    return 0.6745 * (x - med) / mad

# ───────────────────────────────
# Geometry: knee angle
# ───────────────────────────────
def _knee_angle_deg(yx: np.ndarray, hip_i: int, knee_i: int, ankle_i: int) -> np.ndarray:
    a = yx[:, hip_i] - yx[:, knee_i]
    b = yx[:, ankle_i] - yx[:, knee_i]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6
    cos = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def _knee_flexion_deg_from_inner(inner_deg: np.ndarray) -> np.ndarray:
    return 180.0 - np.asarray(inner_deg, dtype=float)

# ───────────────────────────────
# diff_ht 기반 로컬 피크/골짜기
# ───────────────────────────────
def _local_extrema_with_prominence(x: np.ndarray, kind: str, prom_min: float):
    idx = []
    n = len(x)
    for i in range(1, n-1):
        if not np.isfinite(x[i-1:i+2]).all(): continue
        if kind == "max":
            if x[i] >= x[i-1] and x[i] >= x[i+1]:
                left = np.max(x[:i]) if i > 0 else x[i]
                right = np.max(x[i+1:]) if i < n-1 else x[i]
                base = min(left, right)
                prom = x[i] - base
                if prom >= prom_min: idx.append(i)
        else:
            if x[i] <= x[i-1] and x[i] <= x[i+1]:
                left = np.min(x[:i]) if i > 0 else x[i]
                right = np.min(x[i+1:]) if i < n-1 else x[i]
                base = max(left, right)
                prom = base - x[i]
                if prom >= prom_min: idx.append(i)
    return idx

# ───────────────────────────────
# 새 규칙: MS→TO→HS (diff_ht 기반)
# ───────────────────────────────
def _hs_to_ms_single_side(lm_y: np.ndarray, t_ms: np.ndarray,
                          heel_i: int, toe_i: int, hip_i: int, params: GaitParams):
    """반환: (hs_idx, to_idx, ms_idx)  — 기존 키 이름(HS/TO/MS)은 동일하게 사용"""
    heel = smooth1d(lm_y[:, heel_i].astype(float), params.ma_win)
    toe  = smooth1d(lm_y[:, toe_i].astype(float),  params.ma_win)
    diff_ht = heel - toe  # +: heel이 더 낮음(힐 리드)

    # 스케일 및 임계
    scale = np.nanmedian(np.abs(lm_y[:, hip_i] - lm_y[:, heel_i]))
    if not np.isfinite(scale) or scale <= 0:
        s2 = np.nanstd(diff_ht)
        scale = s2 if np.isfinite(s2) and s2 > 0 else 1.0
    eps = params.ms_zero_eps_rel * scale
    prom_min = params.prom_rel * scale

    # [Fail-fast] 전체 진폭이 작으면 이벤트 0건 처리(마비·플랫풋)
    finite = np.isfinite(diff_ht)
    amp95 = np.nanpercentile(np.abs(diff_ht[finite]), 95) if np.any(finite) else 0.0
    if amp95 < params.min_diffht_amp_rel * scale:
        return [], [], []

    # fps / 속도
    if len(t_ms) > 1:
        dt_s = float(np.median(np.diff(t_ms))) / 1000.0
        fps = 1.0 / max(dt_s, 1e-6)
    else:
        fps = 30.0
    v = vel(diff_ht, fps)

    # MS 후보: |diff_ht| 작고 속도도 작은 구간의 중앙 프레임
    v_abs = np.abs(v[np.isfinite(v)])
    v_thr = np.nanpercentile(v_abs, 40) if len(v_abs) else 0.0
    ms_mask = np.isfinite(diff_ht) & (np.abs(diff_ht) <= eps) & (np.abs(v) <= v_thr)
    ms_idx_raw = np.where(ms_mask)[0]
    ms_idx = []
    if len(ms_idx_raw):
        cuts = np.where(np.diff(ms_idx_raw) > 1)[0] + 1
        for seg in np.split(ms_idx_raw, cuts):
            if len(seg) == 0: continue
            ms_idx.append(int(seg[len(seg)//2]))

    # 전 범위 TO/HS 후보(emin/ emax) 미리 계산
    to_cands = _local_extrema_with_prominence(diff_ht, "min", prom_min)
    hs_cands = _local_extrema_with_prominence(diff_ht, "max", prom_min)

    # MS→TO→HS 순서로 묶기
    hs_idx, to_idx, ms_out = [], [], []
    for m in ms_idx:
        # m 이후 TO
        c1 = [j for j in to_cands if j > m and (t_ms[j] - t_ms[m]) >= params.min_gap_ms]
        if not c1: continue
        j = c1[0]
        # TO 이후 HS
        c2 = [k for k in hs_cands if k > j and (t_ms[k] - t_ms[j]) >= params.min_gap_ms]
        if not c2: continue
        k = c2[0]
        to_idx.append(j); hs_idx.append(k); ms_out.append(m)

    L = min(len(hs_idx), len(to_idx), len(ms_out))
    return hs_idx[:L], to_idx[:L], ms_out[:L]

# ───────────────────────────────
# GENU_RECURVATUM(과신전 스냅)
# ───────────────────────────────
def _detect_genu_recurvatum_per_side(
    knee_flex_deg: np.ndarray, hs_list: list[int], to_list: list[int],
    t_ms: np.ndarray, fps: float, params: GaitParams
):
    k = smooth1d(np.asarray(knee_flex_deg, float), win=5)
    omega = np.gradient(k) * fps
    accel = np.gradient(omega) * fps
    jerk  = np.gradient(accel) * fps

    out = []
    n = len(k)
    for hs, to in zip(hs_list, to_list):
        if not (0 <= hs < to <= n-1): continue
        L = to - hs
        if L < 6: continue
        a = hs + int(L * params.stance_mid_lo)
        b = hs + int(L * params.stance_mid_hi)
        if b <= a+2: continue

        zω = _robust_z(omega[a:b]); zj = _robust_z(jerk[a:b])
        thr_ω = np.nanpercentile(omega[a:b], 85)
        idxs = np.arange(a, b)
        is_local_max = (omega[idxs] > np.roll(omega[idxs], 1)) & (omega[idxs] >= np.roll(omega[idxs], -1))
        cond = (
            (k[idxs] <= params.near_ext_flex_deg + 2.0) &
            (zω >= params.z_omega_thresh) &
            ((zj >= params.z_jerk_thresh) | (omega[idxs] > thr_ω)) &
            is_local_max
        )
        cand = idxs[cond]
        if len(cand) == 0: continue

        valid = []
        for i in cand:
            w0 = max(i-2, a); w1 = min(i+3, b)
            acc_seg = accel[w0:w1]
            if len(acc_seg) >= 2 and (acc_seg[0] > 0) and (acc_seg[-1] < 0):
                valid.append(i)
        if len(valid) == 0: continue

        best_i, best_s = None, -1.0
        for i in valid:
            z1 = _robust_z(np.array([omega[i]]))[0]
            z2 = _robust_z(np.array([jerk[i]]))[0]
            bonus = max(0.0, (params.near_ext_flex_deg - k[i]) / max(1e-6, params.near_ext_flex_deg))
            s = z1 + 0.5*z2 + 0.5*bonus
            if s > best_s:
                best_s, best_i = s, i

        out.append({"idx": int(best_i), "time_ms": int(t_ms[best_i]), "score": float(best_s)})
    return out

# ───────────────────────────────
# per-side result builder
# ───────────────────────────────
def _side_result(label: str, t_ms: np.ndarray,
                 hs_idx: list[int], to_idx: list[int], ms_idx: list[int],
                 knee_inner_deg: np.ndarray, lm_y: np.ndarray, toe_i: int,
                 params: GaitParams, fps: float) -> dict:
    knee_flex = _knee_flexion_deg_from_inner(knee_inner_deg)
    qmask = np.isfinite(knee_flex)
    knee_max_inner = float(np.nanmax(knee_inner_deg[qmask])) if np.any(qmask) else 0.0

    near_thr = params.near_ext_flex_deg
    near_mask = qmask & (knee_flex <= near_thr)
    near_idx = np.where(near_mask)[0]
    near_ratio_all = float(near_mask.sum()) / max(1, int(qmask.sum()))
    longest = 0
    if len(near_idx) > 0:
        splits = np.where(np.diff(near_idx) > 1)[0] + 1
        for g in np.split(near_idx, splits):
            longest = max(longest, int(t_ms[g[-1]] - t_ms[g[0]]))

    gr = _detect_genu_recurvatum_per_side(
        knee_flex_deg=knee_flex, hs_list=hs_idx, to_list=to_idx, t_ms=t_ms, fps=fps, params=params
    )
    gr_idx = [e["idx"] for e in gr]
    gr_ms  = [e["time_ms"] for e in gr]
    gr_scores = [e["score"] for e in gr] if gr else []

    swing_peaks, toe_clear = None, None
    if len(hs_idx) >= 1 and len(to_idx) >= 1:
        swing_peaks = []; toe_clear = []
        for i, h in enumerate(hs_idx[:-1]):
            nxt = hs_idx[i+1]
            tos = [t for t in to_idx if h < t < nxt]
            if not tos: continue
            t0 = tos[0]
            seg_knee_flex = knee_flex[t0:nxt+1]
            if len(seg_knee_flex) > 1 and np.all(np.isfinite(seg_knee_flex)):
                swing_peaks.append(float(np.nanmax(seg_knee_flex)))
            seg_toe = lm_y[t0:nxt+1, toe_i]
            if len(seg_toe) > 1 and np.all(np.isfinite(seg_toe)):
                toe_clear.append(float(seg_toe[0] - np.nanmin(seg_toe)))

        if swing_peaks == []: swing_peaks = None
        if toe_clear == []: toe_clear = None

    metrics_core = {
        "knee_max_inner_deg": round(knee_max_inner, 2),
        "near_ext_ratio_all": round(near_ratio_all, 3),
        "near_ext_longest_ms": int(longest),
        "near_ext_threshold_flex_deg": near_thr,
        "stiff_knee_flag": bool(swing_peaks is not None and np.mean(swing_peaks) < params.stiff_knee_deg),
        "swing_peak_flex_mean_deg": float(np.mean(swing_peaks)) if swing_peaks else None,
        "toe_clear_mean": float(np.mean(toe_clear)) if toe_clear else None,
        "genu_recurvatum_count": int(len(gr_ms)),
        "genu_recurvatum_score_max": float(np.max(gr_scores)) if gr_scores else None,
    }

    return {
        "side": label,
        "events": {
            "HS_idx": hs_idx, "TO_idx": to_idx, "MS_idx": ms_idx,
            "HS_ms": [int(t_ms[i]) for i in hs_idx],
            "TO_ms": [int(t_ms[i]) for i in to_idx],
            "MS_ms": [int(t_ms[i]) for i in ms_idx],
            "GENU_RECURVATUM_idx": gr_idx,
            "GENU_RECURVATUM_ms": gr_ms,
        },
        "metrics": metrics_core,
    }

# ───────────────────────────────
# Bilateral API
# ───────────────────────────────
def detect_events_bilateral(npz_path: str, params: GaitParams | None = None) -> dict:
    params = params or GaitParams()
    D = load_npz(npz_path)
    lx, ly, t_ms = D["lm_x"], D["lm_y"], D["t_ms"]
    valid, meta = D["valid"], D["meta"]
    vmean, bbr, qok = D["vis_mean"], D["bbox_ratio"], D["quality_ok"]

    # fps
    if len(t_ms) > 1:
        dt_s = float(np.median(np.diff(t_ms))) / 1000.0
        fps = 1.0 / max(dt_s, 1e-6)
    else:
        fps = float(meta.get("fps", 30))

    # 품질 가드
    mask = valid.copy()
    if vmean is not None: mask &= (vmean >= params.vis_min)
    if bbr   is not None: mask &= (bbr >= params.bbox_min) & (bbr <= params.bbox_max)
    if qok   is not None: mask &= qok.astype(bool)
    lx = lx.copy(); ly = ly.copy()
    lx[~mask, :] = np.nan; ly[~mask, :] = np.nan

    # 새 규칙 HS/TO/MS
    hs_L, to_L, ms_L = _hs_to_ms_single_side(ly, t_ms, L_HEEL, L_FOOT_INDEX, L_HIP, params)
    hs_R, to_R, ms_R = _hs_to_ms_single_side(ly, t_ms, R_HEEL, R_FOOT_INDEX, R_HIP, params)

    # 각도
    yx = np.stack([ly, lx], axis=-1)
    knee_inner_L = _knee_angle_deg(yx, L_HIP, L_KNEE, L_ANKLE)
    knee_inner_R = _knee_angle_deg(yx, R_HIP, R_KNEE, R_ANKLE)

    left_res  = _side_result("LEFT",  t_ms, hs_L, to_L, ms_L, knee_inner_L, ly, L_FOOT_INDEX, params, fps)
    right_res = _side_result("RIGHT", t_ms, hs_R, to_R, ms_R, knee_inner_R, ly, R_FOOT_INDEX, params, fps)

    return {
        "task": "gait",
        "npz": str(npz_path),
        "meta": meta,
        "params": vars(params),
        "LEFT": left_res,
        "RIGHT": right_res,
    }

# ───────────────────────────────
# STS API
# ───────────────────────────────
def detect_sts_events(npz_path: str, params: STSParams | None = None) -> dict:
    params = params or STSParams()
    D = load_npz(npz_path)
    lx, ly, t_ms = D["lm_x"], D["lm_y"], D["t_ms"]
    meta = D["meta"]

    dt = np.median(np.diff(t_ms)) / 1000.0 if len(t_ms) > 1 else 1 / max(meta.get("fps", 30), 1)
    pelvis_y = smooth1d((ly[:, L_HIP] + ly[:, R_HIP]) / 2.0, params.ma_win)
    yx = np.stack([ly, lx], axis=-1)
    knee_deg_l = _knee_angle_deg(yx, L_HIP, L_KNEE, L_ANKLE)
    knee_deg_r = _knee_angle_deg(yx, R_HIP, R_KNEE, R_ANKLE)
    knee_deg = smooth1d((knee_deg_l + knee_deg_r) / 2.0, params.ma_win)

    v = diff(pelvis_y, dt)

    # seat-off
    so_idx, last_t = [], -1e9
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] < v[i + 1] and v[i] < params.v_pelvis_drop:
            if (t_ms[i] - last_t) > params.min_cycle_ms:
                so_idx.append(i); last_t = t_ms[i]

    # full-stand
    fs_idx = []
    kv = diff(knee_deg, dt)
    for s in so_idx:
        for j in range(s + 1, len(v)):
            if abs(v[j]) < params.v_pelvis_stop and (j - s) > int(0.5 / dt) and abs(kv[j]) < params.kv_knee_stop:
                fs_idx.append(j); break

    cycles = min(len(so_idx), len(fs_idx))
    durations = [(t_ms[fs_idx[i]] - t_ms[so_idx[i]]) / 1000.0 for i in range(cycles)]

    return {
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

# ───────────────────────────────
# Save helpers
# ───────────────────────────────
def save_events_json(result: dict, out_path: str | Path): _export_json(result, out_path)

def _timeline_rows_gait(result: dict):
    rows = []
    for side in ["LEFT", "RIGHT"]:
        ev = result.get(side, {}).get("events", {})
        for k in ["HS_ms", "TO_ms", "MS_ms", "GENU_RECURVATUM_ms"]:
            for t in ev.get(k, []):
                rows.append({"side": side, "event": k, "time_ms": int(t)})
    rows.sort(key=lambda r: r["time_ms"])
    for i, r in enumerate(rows):
        r["order"] = i
    return rows

def save_events_csv_timeline(result: dict, out_csv: str | Path):
    if result.get("task") == "gait":
        rows = _timeline_rows_gait(result)
    else:
        rows = []
        ev = result.get("events", {})
        for k in ["seat_off_ms", "full_stand_ms"]:
            for t in ev.get(k, []):
                rows.append({"event": k, "time_ms": int(t)})
        rows.sort(key=lambda r: r["time_ms"])
        for i, r in enumerate(rows):
            r["order"] = i
    _export_csv(rows, out_csv)

def save_events_csv_metrics(result: dict, out_csv: str | Path):
    rows = []
    if result.get("task") == "gait":
        for side in ["LEFT", "RIGHT"]:
            side_res = result.get(side, {})
            m  = side_res.get("metrics", {})
            ev = side_res.get("events",  {})
            rows.append({
                "side": side,
                # 이벤트 개수
                "hs_count": len(ev.get("HS_ms", [])),
                "to_count": len(ev.get("TO_ms", [])),
                "ms_count": len(ev.get("MS_ms", [])),
                "genu_recurvatum_count": m.get("genu_recurvatum_count"),
                # 핵심 지표
                "knee_max_inner_deg": m.get("knee_max_inner_deg"),
                "near_ext_ratio_all": m.get("near_ext_ratio_all"),
                "near_ext_longest_ms": m.get("near_ext_longest_ms"),
                "near_ext_threshold_flex_deg": m.get("near_ext_threshold_flex_deg"),
                "stiff_knee_flag": m.get("stiff_knee_flag"),
                "swing_peak_flex_mean_deg": m.get("swing_peak_flex_mean_deg"),
                "toe_clear_mean": m.get("toe_clear_mean"),
                "genu_recurvatum_score_max": m.get("genu_recurvatum_score_max"),
                # 참고 타임스탬프
                "hs_ms_list": ";".join(str(x) for x in ev.get("HS_ms", [])),
                "to_ms_list": ";".join(str(x) for x in ev.get("TO_ms", [])),
                "ms_ms_list": ";".join(str(x) for x in ev.get("MS_ms", [])),
                "genu_recurvatum_ms_list": ";".join(strx for strx in ev.get("GENU_RECURVATUM_ms", [])),
            })
    else:
        m = result.get("metrics", {})
        rows.append({
            "cycles": m.get("cycles"),
            "mean_cycle_sec": m.get("mean_cycle_sec"),
        })
    _export_csv(rows, out_csv)

# ───────────────────────────────
# Steps / Strides builders
# ───────────────────────────────
def _hs_events_chrono(result: dict):
    evs = []
    for side in ["LEFT", "RIGHT"]:
        for t in result.get(side, {}).get("events", {}).get("HS_ms", []):
            evs.append({"side": side, "time_ms": int(t)})
    evs.sort(key=lambda r: r["time_ms"])
    return evs

def save_steps_csv(result: dict, out_csv: str | Path):
    rows = []
    if result.get("task") != "gait":
        _export_csv(rows, out_csv); return
    ev = _hs_events_chrono(result)
    step_id = 0
    for i in range(len(ev)-1):
        a, b = ev[i], ev[i+1]
        if a["side"] != b["side"]:
            step_id += 1
            rows.append({
                "step_id": step_id,
                "lead_side": a["side"],
                "hs_lead_ms": a["time_ms"],
                "hs_contra_ms": b["time_ms"],
                "step_time_ms": b["time_ms"] - a["time_ms"],
            })
    _export_csv(rows, out_csv)

def save_strides_csv(result: dict, out_csv: str | Path):
    rows = []
    if result.get("task") != "gait":
        _export_csv(rows, out_csv); return
    ev = _hs_events_chrono(result)
    last_by_side = {}
    stride_id = 0
    for e in ev:
        s = e["side"]; t = e["time_ms"]
        if s in last_by_side:
            stride_id += 1
            rows.append({
                "stride_id": stride_id,
                "side": s,
                "hs_ms": last_by_side[s],
                "hs_next_ms": t,
                "stride_time_ms": t - last_by_side[s],
            })
        last_by_side[s] = t
    _export_csv(rows, out_csv)

# ───────────────────────────────
# Print summaries
# ───────────────────────────────
def _print_summary_gait(res: dict):
    for side in ["LEFT", "RIGHT"]:
        ev = res.get(side, {}).get("events", {})
        m  = res.get(side, {}).get("metrics", {})
        print(f"[{side}] HS n={len(ev.get('HS_ms', []))}, TO n={len(ev.get('TO_ms', []))}, "
              f"MS n={len(ev.get('MS_ms', []))}, GENU_RECURVATUM n={len(ev.get('GENU_RECURVATUM_ms', []))}")
        print(f"      knee_max_inner={m.get('knee_max_inner_deg',0):.1f}°, "
              f"near-ext ratio={m.get('near_ext_ratio_all',0):.3f}, "
              f"GR count={m.get('genu_recurvatum_count',0)}")

def _print_summary_sts(res: dict):
    m = res.get("metrics", {})
    ev = res.get("events", {})
    print("STS:")
    print(f"  seat-off n={len(ev.get('seat_off_ms', []))}, full-stand n={len(ev.get('full_stand_ms', []))}")
    print(f"  cycles={m.get('cycles',0)}, mean_cycle_sec={m.get('mean_cycle_sec',0)}")

# ───────────────────────────────
# CLI
# ───────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Gait and STS events from pose npz")
    p.add_argument("--task", choices=["gait", "sts"], default="gait")
    p.add_argument("--npz", required=True)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    a = p.parse_args()

    stem = Path(a.npz).stem
    outdir = Path("results/reports"); outdir.mkdir(parents=True, exist_ok=True)

    if a.task == "gait":
        res = detect_events_bilateral(a.npz)
        _print_summary_gait(res)
        stem += "_bilateral"
    else:
        res = detect_sts_events(a.npz)
        _print_summary_sts(res)
        stem += "_sts"

    if a.save_json:
        save_events_json(res, outdir / f"{stem}.json")
        print(f"[saved] results/reports/{stem}.json")

    if a.save_csv:
        save_events_csv_timeline(res, outdir / f"{stem}_timeline.csv")
        print(f"[saved] results/reports/{stem}_timeline.csv")
        save_events_csv_metrics(res, outdir / f"{stem}_metrics.csv")
        print(f"[saved] results/reports/{stem}_metrics.csv")
        if res.get("task") == "gait":
            save_steps_csv(res, outdir / f"{stem}_steps.csv")
            print(f"[saved] results/reports/{stem}_steps.csv")
            save_strides_csv(res, outdir / f"{stem}_strides.csv")
            print(f"[saved] results/reports/{stem}_strides.csv")
