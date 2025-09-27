# -*- coding: utf-8 -*-
"""
파일명: src/events.py

설명:
  - Mediapipe Pose 시계열(.npz)에서 보행(Gait) 이벤트/지표 계산.
  - HS/TO/MS: diff_ht(t)=heel_y−toe_y 기반 완화 규칙.
  - GR(과신전): MS±창 내에서 (무릎 내부각 ≥ gr_angle_inner_deg) ∧ (knee_x 부호전환 구간 존재).
  - 산출: 이벤트(HS, TO, MS, GENU_RECURVATUM) + 지표(무릎 최대/최소 내부각, near-extension 비율/최대 지속, stiff-knee 개수)
  - 저장: JSON/CSV

입력: pose npz {lm_x,lm_y,lm_v,t_ms,valid,meta,...}
주의: y축은 이미지 좌표계(아래가 +).
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
    # 핵심 평활·임계
    ma_win: int = 7                   # 이동평균 창
    ms_zero_eps_rel: float = 0.02     # MS: |heel−toe|≤eps 비율
    min_gap_ms: int = 120             # 이벤트 간 최소 간격
    min_diffht_amp_rel: float = 0.003 # diff_ht 진폭 없을 때 스킵

    # near-extension
    ms_plateau_min_ms: float = 80.0
    near_ext_flex_deg: float = 8.0
    ms_border_deg: float = 178.0
    ms_def_deg: float = 180.0

    # ─ GR(각도+부호전환) 규칙 ─
    gr_angle_inner_deg: float = 175.0   # 내부각 임계
    gr_ms_window_ms: float = 400.0      # MS 주변 창(±)
    snap_min_dur_ms: float = 40.0       # 부호전환 연속 최소 지속(ms)
    flip_min_disp: float = 0.01         # 부호전환 구간 누적 |dx| 최소(노이즈 컷)

# ───────────────────────────────
# IO / math utils
# ───────────────────────────────
def load_npz(path: str | Path):
    d = np.load(path, allow_pickle=True)
    meta_raw = d["meta"] if "meta" in d.files else "{}"
    try:
        meta = json.loads(str(meta_raw.item() if hasattr(meta_raw, "item") else meta_raw))
    except Exception:
        meta = {}
    def opt(name): return d[name] if name in d.files else None
    return {
        "lm_x": d["lm_x"], "lm_y": d["lm_y"], "lm_v": opt("lm_v"),
        "t_ms": d["t_ms"] if "t_ms" in d.files else np.arange(d["lm_x"].shape[0]) * 1000/30.0,
        "valid": d["valid"].astype(bool) if "valid" in d.files else np.ones(d["lm_x"].shape[0], bool),
        "vis_mean": opt("vis_mean"), "n_visible": opt("n_visible"),
        "bbox_ratio": opt("bbox_ratio"), "jitter": opt("jitter"),
        "quality_ok": opt("quality_ok"), "meta": meta,
    }

def smooth1d(x: np.ndarray, win: int):
    if win <= 1: return x
    k = np.ones(win, dtype=float) / win
    return np.convolve(x.astype(float), k, mode="same")

def _export_json(obj: dict, out: str | Path):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    def _to_py(x):
        import numpy as _np
        if isinstance(x, (_np.integer,)): return int(x)
        if isinstance(x, (_np.floating,)): return float(x)
        if isinstance(x, (_np.bool_,)): return bool(x)
        if isinstance(x, (_np.ndarray,)): return _np.nan_to_num(x, nan=None).tolist()
        if isinstance(x, dict): return {k: _to_py(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [_to_py(v) for v in x]
        return x
    with out.open("w", encoding="utf-8") as f:
        json.dump(_to_py(obj), f, ensure_ascii=False, indent=2)

def _export_csv(rows: list[dict], out: str | Path):
    if pd is None: return
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

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
def _local_extrema_simple(x: np.ndarray, kind: str):
    if len(x) < 3: return np.array([], int)
    if kind == "max":
        return np.where((x[1:-1] >= x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    else:
        return np.where((x[1:-1] <= x[:-2]) & (x[1:-1] <= x[2:]))[0] + 1

# ───────────────────────────────
# 새 규칙: MS→TO→HS (diff_ht + 무릎각)
# ───────────────────────────────
def _hs_to_ms_single_side_with_knee(
    lm_y: np.ndarray, t_ms: np.ndarray,
    heel_i: int, toe_i: int, hip_i: int,
    knee_inner: np.ndarray, params: GaitParams
):
    heel = smooth1d(lm_y[:, heel_i].astype(float), max(3, params.ma_win))
    toe  = smooth1d(lm_y[:, toe_i ].astype(float), max(3, params.ma_win))
    diff_ht = heel - toe
    knee = smooth1d(np.asarray(knee_inner, float), max(5, params.ma_win))

    scale = np.nanmedian(np.abs(lm_y[:, hip_i] - lm_y[:, heel_i]))
    if not np.isfinite(scale) or scale <= 0:
        s2 = np.nanstd(diff_ht); scale = s2 if np.isfinite(s2) and s2>0 else 1.0
    eps = max(0.5*params.ms_zero_eps_rel, params.ms_zero_eps_rel) * scale
    min_gap = int(params.min_gap_ms*0.7)
    n = len(diff_ht)
    if n < 5: return [], [], []

    sign = np.sign(diff_ht)
    kmax = _local_extrema_simple(knee, "max")
    # MS 후보
    ms_zone = np.where(np.isfinite(diff_ht) & (np.abs(diff_ht) <= eps))[0]
    ms_cands = sorted(set(ms_zone).intersection(set(kmax)))
    if not ms_cands:
        zc = np.where(np.sign(diff_ht[:-1]) * np.sign(diff_ht[1:]) < 0)[0]
        for i in zc:
            a = max(0, i-5); b = min(n, i+6)
            j = a + int(np.nanargmax(knee[a:b]))
            ms_cands.append(j)
        ms_cands = sorted(set(ms_cands))

    def next_in_window(start_ms, want_pos: bool, radius_ms: int = 800):
        s = int(np.searchsorted(t_ms, start_ms + min_gap, 'left'))
        e = int(np.searchsorted(t_ms, start_ms + radius_ms, 'right'))
        s = max(1, s); e = min(n-1, e)
        if e <= s+1: return None
        if want_pos:
            mask = np.where(sign[s:e] > 0)[0]
            if mask.size == 0: return None
            idx = s + mask
            return int(idx[np.nanargmax(knee[idx])])
        else:
            mask = np.where(sign[s:e] < 0)[0]
            if mask.size == 0: return None
            idx = s + mask
            return int(idx[np.nanargmin(knee[idx])])

    hs_idx, to_idx, ms_idx = [], [], []
    for m in ms_cands:
        if m not in kmax:
            a = max(0, m-5); b = min(n, m+6)
            m = a + int(np.nanargmax(knee[a:b]))
        j = next_in_window(t_ms[m], want_pos=False)
        if j is None: continue
        k = next_in_window(t_ms[j], want_pos=True)
        if k is None: continue
        ms_idx.append(m); to_idx.append(j); hs_idx.append(k)

    L = min(len(hs_idx), len(to_idx), len(ms_idx))
    return hs_idx[:L], to_idx[:L], ms_idx[:L]

# ───────────────────────────────
# GR 부호전환 기반 유틸
# ───────────────────────────────
def _dominant_sign(dx: np.ndarray) -> int:
    """dx의 지배 부호(+1 또는 -1). 0 근처는 제외하고 다수결."""
    x = dx[np.isfinite(dx)]
    x = x[np.abs(x) > 1e-6]
    if x.size == 0:
        return 0
    pos = np.sum(x > 0)
    neg = np.sum(x < 0)
    if pos == neg: return 0
    return 1 if pos > neg else -1

def _sign_flip_spans(knee_x: np.ndarray, t_ms: np.ndarray,
                     min_dur_ms: float,
                     min_disp: float,
                     smooth_win: int = 5):
    """
    knee_x 차분 dx의 지배부호와 반대 부호가 연속되는 '부호전환' 구간 반환.
    반환: list[[t_start, t_end, t_mid, disp]]
    """
    x = knee_x.astype(float).copy()
    if smooth_win and smooth_win > 1:
        k = np.ones(smooth_win, float)/smooth_win
        x = np.convolve(x, k, mode="same")
    dx = np.diff(x, prepend=x[0])
    dom = _dominant_sign(dx)
    if dom == 0:
        return []  # 지배 부호 결정 불가(정체 등)

    opp = -dom
    mask = np.sign(dx) == opp

    spans = []
    n = len(dx); i = 0
    while i < n:
        if not mask[i]:
            i += 1; continue
        j = i
        while j < n and mask[j]:
            j += 1
        t0 = float(t_ms[max(i-1, 0)])
        t1 = float(t_ms[min(j, n-1)])
        if (t1 - t0) >= min_dur_ms:
            disp = float(np.nansum(np.abs(dx[i:j])))
            if disp >= min_disp:
                spans.append([t0, t1, (t0+t1)/2.0, disp])
        i = j
    return spans

# ───────────────────────────────
# GR: knee_x 부호전환 + 각도 임계
# ───────────────────────────────
def _detect_genu_recurvatum_per_side(
    knee_inner_deg: np.ndarray,
    knee_x: np.ndarray,
    t_ms: np.ndarray,
    ms_idx: list[int],
    params: GaitParams
):
    """
    GR = (MS±window 내) AND (무릎 내부각 최대 ≥ gr_angle_inner_deg) AND (knee_x 부호전환 구간 존재)
    방향 무관.
    """
    out = []
    if len(ms_idx) == 0: return out

    k_in = np.asarray(knee_inner_deg, float)
    x    = np.asarray(knee_x, float)
    t    = np.asarray(t_ms, float)

    # 전 구간 부호전환 후보
    flip_spans = _sign_flip_spans(
        x, t,
        min_dur_ms=max(20.0, float(params.snap_min_dur_ms)),
        min_disp=float(params.flip_min_disp),
        smooth_win=5
    )

    for m in ms_idx:
        t0 = t[m] - params.gr_ms_window_ms
        t1 = t[m] + params.gr_ms_window_ms
        a = int(np.searchsorted(t, t0, 'left'))
        b = int(np.searchsorted(t, t1, 'right'))
        a = max(0, a); b = min(len(t), b)
        if b - a < 3: continue

        # 각도 조건
        if np.nanmax(k_in[a:b]) < params.gr_angle_inner_deg:
            continue

        # 윈도우 내 부호전환 겹침 여부
        cand = [s for s in flip_spans if not (s[1] < t0 or s[0] > t1)]
        if not cand:
            continue

        s0 = sorted(cand, key=lambda z: z[2])[0]
        out.append({
            "ms_idx": int(m),
            "time_ms": int(round(s0[2])),
            "span_start_ms": int(round(s0[0])),
            "span_end_ms": int(round(s0[1])),
            "flip_disp": float(s0[3]),
            "angle_max_inner_ms_win": float(np.nanmax(k_in[a:b]))
        })
    return out

# ───────────────────────────────
# Stiff knee (TO 시점 내부각 기준)
# ───────────────────────────────
def detect_stiff_knee_at_to(
    knee_inner: np.ndarray,
    t_ms: np.ndarray,
    to_idx: list[int],
    thresh_inner_deg: float = 140.0
):
    ki = np.asarray(knee_inner, float)
    t  = np.asarray(t_ms, float)
    out = []
    for j in to_idx:
        if not (0 <= j < len(ki)) or not np.isfinite(ki[j]):
            continue
        inner = float(ki[j])
        flex  = float(180.0 - inner)
        stiff = bool(inner > thresh_inner_deg)
        out.append({
            "to_ms": int(t[j]),
            "knee_inner": round(inner, 2),
            "knee_flexion": round(flex, 2),
            "stiff": stiff
        })
    return out

# ───────────────────────────────
# per-side result builder
# ───────────────────────────────
def _side_result(label: str, t_ms: np.ndarray,
                 hs_idx: list[int], to_idx: list[int], ms_idx: list[int],
                 knee_inner_deg: np.ndarray, knee_x: np.ndarray,
                 lm_y: np.ndarray, toe_i: int,
                 params: GaitParams, fps: float) -> dict:

    knee_flex = _knee_flexion_deg_from_inner(knee_inner_deg)
    qmask = np.isfinite(knee_flex)

    knee_max_inner = float(np.nanmax(knee_inner_deg[qmask])) if np.any(qmask) else 0.0
    knee_min_inner = float(np.nanmin(knee_inner_deg[qmask])) if np.any(qmask) else 180.0

    near_thr = params.near_ext_flex_deg
    near_mask = qmask & (knee_flex <= near_thr)
    near_idx = np.where(near_mask)[0]
    near_ratio_all = float(near_mask.sum()) / max(1, int(qmask.sum()))
    longest = 0
    if len(near_idx) > 0:
        splits = np.where(np.diff(near_idx) > 1)[0] + 1
        for g in np.split(near_idx, splits):
            longest = max(longest, int(t_ms[g[-1]] - t_ms[g[0]]))

    # GR: 각도+부호전환 복합
    gr_items = _detect_genu_recurvatum_per_side(
        knee_inner_deg=knee_inner_deg, knee_x=knee_x, t_ms=t_ms, ms_idx=ms_idx, params=params
    )
    gr_ms_all = [int(e["time_ms"]) for e in gr_items]

    # Stiff-knee
    sk_items = detect_stiff_knee_at_to(knee_inner=knee_inner_deg, t_ms=t_ms, to_idx=to_idx, thresh_inner_deg=140.0)
    stiff_cnt = int(sum(1 for it in sk_items if it.get("stiff", False)))

    metrics_core = {
        "knee_max_inner_deg": round(knee_max_inner, 2),
        "knee_min_inner_deg": round(knee_min_inner, 2),
        "near_ext_ratio_all": round(near_ratio_all, 3),
        "near_ext_longest_ms": int(longest),
        "stiff_knee_count": stiff_cnt,
    }

    return {
        "side": label,
        "events": {
            "HS_idx": hs_idx, "TO_idx": to_idx, "MS_idx": ms_idx,
            "HS_ms": [int(t_ms[i]) for i in hs_idx],
            "TO_ms": [int(t_ms[i]) for i in to_idx],
            "MS_ms": [int(t_ms[i]) for i in ms_idx],
            "GENU_RECURVATUM_ms": sorted(gr_ms_all),
        },
        "metrics": metrics_core,
        "stiff_knee_items": sk_items,
        "gr_items": gr_items,  # 부호전환/각도 상세
    }

# ───────────────────────────────
# Bilateral API
# ───────────────────────────────
def detect_events_bilateral(npz_path: str, params: GaitParams | None = None) -> dict:
    params = params or GaitParams()
    D = load_npz(npz_path)
    lx, ly, t_ms = D["lm_x"], D["lm_y"], D["t_ms"]
    valid, meta = D["valid"], D["meta"]

    # fps
    if len(t_ms) > 1:
        dt_s = float(np.median(np.diff(t_ms))) / 1000.0
        fps = 1.0 / max(dt_s, 1e-6)
    else:
        fps = float(meta.get("fps", 30))

    # 각도
    yx = np.stack([ly, lx], axis=-1)  # (y,x)
    knee_inner_L = _knee_angle_deg(yx, L_HIP, L_KNEE, L_ANKLE)
    knee_inner_R = _knee_angle_deg(yx, R_HIP, R_KNEE, R_ANKLE)

    # 무릎 x좌표
    knee_x_L = lx[:, L_KNEE].astype(float)
    knee_x_R = lx[:, R_KNEE].astype(float)

    # HS/TO/MS
    hs_L, to_L, ms_L = _hs_to_ms_single_side_with_knee(
        lm_y=ly, t_ms=t_ms, heel_i=L_HEEL, toe_i=L_FOOT_INDEX, hip_i=L_HIP,
        knee_inner=knee_inner_L, params=params
    )
    hs_R, to_R, ms_R = _hs_to_ms_single_side_with_knee(
        lm_y=ly, t_ms=t_ms, heel_i=R_HEEL, toe_i=R_FOOT_INDEX, hip_i=R_HIP,
        knee_inner=knee_inner_R, params=params
    )

    left_res  = _side_result("LEFT",  t_ms, hs_L, to_L, ms_L, knee_inner_L, knee_x_L, ly, L_FOOT_INDEX, params, fps)
    right_res = _side_result("RIGHT", t_ms, hs_R, to_R, ms_R, knee_inner_R, knee_x_R, ly, R_FOOT_INDEX, params, fps)

    return {
        "task": "gait",
        "npz": str(npz_path),
        "meta": meta,
        "params": vars(params),
        "LEFT": left_res,
        "RIGHT": right_res,
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
    rows = _timeline_rows_gait(result)
    _export_csv(rows, out_csv)

def save_events_csv_metrics(result: dict, out_csv: str | Path):
    rows = []
    for side in ["LEFT", "RIGHT"]:
        side_res = result.get(side, {})
        m  = side_res.get("metrics", {})
        ev = side_res.get("events",  {})
        rows.append({
            "side": side,
            "hs_count": len(ev.get("HS_ms", [])),
            "to_count": len(ev.get("TO_ms", [])),
            "ms_count": len(ev.get("MS_ms", [])),
            "gr_count": len(ev.get("GENU_RECURVATUM_ms", [])),
            "stiff_count": m.get("stiff_knee_count"),
            "knee_max_inner_deg": m.get("knee_max_inner_deg"),
            "knee_min_inner_deg": m.get("knee_min_inner_deg"),
            "near_ext_ratio_all": m.get("near_ext_ratio_all"),
            "near_ext_longest_ms": m.get("near_ext_longest_ms"),
            "hs_ms_list": ";".join(str(x) for x in ev.get("HS_ms", [])),
            "to_ms_list": ";".join(str(x) for x in ev.get("TO_ms", [])),
            "ms_ms_list": ";".join(str(x) for x in ev.get("MS_ms", [])),
            "gr_ms_list": ";".join(str(x) for x in ev.get("GENU_RECURVATUM_ms", [])),
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
              f"MS n={len(ev.get('MS_ms', []))}, GR n={len(ev.get('GENU_RECURVATUM_ms', []))}, "
              f"SK n={m.get('stiff_knee_count', 0)}")
        print(f"      knee_max_inner={m.get('knee_max_inner_deg',0):.1f}°,  "
              f"knee_min_inner={m.get('knee_min_inner_deg',180):-.1f}°")

# ───────────────────────────────
# CLI
# ───────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Gait events from pose npz")
    p.add_argument("--npz", required=True)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    a = p.parse_args()

    stem = Path(a.npz).stem + "_bilateral"
    outdir = Path("results/reports"); outdir.mkdir(parents=True, exist_ok=True)

    res = detect_events_bilateral(a.npz)
    _print_summary_gait(res)

    if a.save_json:
        save_events_json(res, outdir / f"{stem}.json")
        print(f"[saved] results/reports/{stem}.json")

    if a.save_csv:
        save_events_csv_timeline(res, outdir / f"{stem}_timeline.csv")
        print(f"[saved] results/reports/{stem}_timeline.csv")
        save_events_csv_metrics(res, outdir / f"{stem}_metrics.csv")
        print(f"[saved] results/reports/{stem}_metrics.csv")
        save_steps_csv(res, outdir / f"{stem}_steps.csv")
        print(f"[saved] results/reports/{stem}_steps.csv")
        save_strides_csv(res, outdir / f"{stem}_strides.csv")
        print(f"[saved] results/reports/{stem}_strides.csv")
