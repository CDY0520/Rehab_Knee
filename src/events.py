"""
파일명: src/events.py
설명:
  - Mediapipe Pose 시계열(.npz) 입력을 이용하여 보행(Gait) 및 STS(앉았다 일어서기) 이벤트와 지표를 계산한다.
  - Gait:
      · 이벤트: HS(Heel Strike), TO(Toe Off), MS(Mid Stance: contralateral TO→HS 구간의 골반 y 최소)
      · 지표: 무릎 최대각, 과신전 비율/최대 지속시간, stiff-knee, toe-clearance
  - STS:
      · 이벤트: Seat-off, Full-stand
      · 지표: 사이클 횟수, 평균 소요시간
  - 품질 지표(quality_ok, vis_mean, bbox_ratio 등) 적용 가능

블록 구성:
  0) 임포트/상수
  1) 공통 유틸(npz 로딩, 스무딩, 미분, 내보내기)
  2) Gait 이벤트 검출(HS/TO/MS) + 무릎 지표
  3) STS 이벤트 검출(seat-off, full-stand) + 지표
  4) 저장 API(JSON/CSV)
  5) CLI: --task gait / --task sts

사용 예:
  보행: python src/events.py --task gait --npz results/keypoints/sample_walk.npz --save-json --save-csv
  STS : python src/events.py --task sts  --npz results/keypoints/sample_sts.npz  --save-json --save-csv
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

# ── Mediapipe landmark indices
L_ANKLE, R_ANKLE = 27, 28
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

# ── params
@dataclass
class GaitParams:
    ma_win: int = 3
    min_step_interval_ms: int = 200
    vis_min: float = 0.6
    bbox_min: float = 0.01
    bbox_max: float = 0.6
    hyperext_deg: float = 185.0
    stiff_knee_deg: float = 40.0
    toe_clear_min: float = 0.012

@dataclass
class STSParams:
    ma_win: int = 7
    min_cycle_ms: int = 1500         # seat-off 최소 간격
    v_pelvis_drop: float = -0.02     # seat-off 후보(하강 속도)
    v_pelvis_stop: float = 0.005     # full-stand 정지 임계
    kv_knee_stop: float = 5.0        # full-stand 무릎 각속도 정지 임계(°/s)

# ── io / utils
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

def _export_json(obj: dict, out: str | Path):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _export_csv(rows: list[dict], out: str | Path):
    if pd is None: return
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8-sig")

# ── geometry
def _knee_angle_deg(yx: np.ndarray, hip_i: int, knee_i: int, ankle_i: int) -> np.ndarray:
    a = yx[:, hip_i] - yx[:, knee_i]
    b = yx[:, ankle_i] - yx[:, knee_i]
    num = (a * b).sum(axis=1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-6
    cos = np.clip(num / den, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

# ── gait: single-side HS/TO
def _hs_to_single_side(lm_y: np.ndarray, t_ms: np.ndarray,
                       ankle_i: int, toe_i: int,
                       ma_win: int, min_step_ms: int):
    # 아래로 갈수록 값↑ → 접지 높이 = max(ankle, toe)
    foot_y = np.maximum(lm_y[:, ankle_i], lm_y[:, toe_i]).copy()
    sig = smooth1d(foot_y, ma_win)

    # HS: 로컬 최대(간격 제약)
    hs_idx = []
    last_t = -1e9
    for i in range(1, len(sig)-1):
        if np.isfinite(sig[i]) and sig[i] >= sig[i-1] and sig[i] >= sig[i+1]:
            if (t_ms[i] - last_t) >= min_step_ms:
                hs_idx.append(i); last_t = t_ms[i]

    # TO: 연속 HS 사이 로컬 최소 1개 선택
    to_idx = []
    for k, h in enumerate(hs_idx[:-1]):
        nxt = hs_idx[k+1]
        jmin = None; vmin = np.inf
        for j in range(max(h+1,1), min(nxt, len(sig)-1)):
            if np.isfinite(sig[j]) and sig[j] <= sig[j-1] and sig[j] <= sig[j+1]:
                if sig[j] < vmin: vmin = sig[j]; jmin = j
        if jmin is not None:
            to_idx.append(jmin)

    return hs_idx, to_idx

# ── gait: contralateral TO→HS → MS
def _ms_from_contra(pelvis_y: np.ndarray, to_ctr: list[int], hs_ctr: list[int]) -> list[int]:
    ms = []; j = 0
    for to in to_ctr:
        while j < len(hs_ctr) and hs_ctr[j] <= to: j += 1
        if j >= len(hs_ctr): break
        hs = hs_ctr[j]
        if hs - to >= 2:
            seg = pelvis_y[to:hs+1]
            ms.append(int(np.nanargmin(seg) + to))
    return ms

# ── gait: side result
def _side_result(label: str, t_ms: np.ndarray, hs_idx: list[int], to_idx: list[int], ms_idx: list[int],
                 knee_deg: np.ndarray, lm_y: np.ndarray, toe_i: int, params: GaitParams) -> dict:
    qmask = np.isfinite(knee_deg)
    knee_max = float(np.nanmax(knee_deg[qmask])) if np.any(qmask) else 0.0
    thr = params.hyperext_deg
    he_mask = qmask & (knee_deg > thr)
    he_idx = np.where(he_mask)[0]
    he_ratio_all = float(he_mask.sum()) / max(1, int(qmask.sum()))
    longest = 0
    if len(he_idx) > 0:
        splits = np.where(np.diff(he_idx) > 1)[0] + 1
        for g in np.split(he_idx, splits):
            longest = max(longest, int(t_ms[g[-1]] - t_ms[g[0]]))

    metrics_knee_only = {
        "knee_max_deg": round(knee_max, 2),
        "hyperext_ratio_all": round(he_ratio_all, 3),
        "hyperext_longest_ms": int(longest),
        "hyperext_threshold_deg": thr,
    }

    swing_peaks, toe_clear = None, None
    if len(hs_idx) >= 1 and len(to_idx) >= 1:
        swing_peaks = []; toe_clear = []
        for i, h in enumerate(hs_idx[:-1]):
            nxt = hs_idx[i+1]
            tos = [t for t in to_idx if h < t < nxt]
            if not tos: continue
            t = tos[0]
            seg_knee = knee_deg[t:nxt+1]
            if len(seg_knee) > 1 and np.all(np.isfinite(seg_knee)):
                swing_peaks.append(float(np.nanmax(seg_knee)))
            seg_toe = lm_y[t:nxt+1, toe_i]
            if len(seg_toe) > 1 and np.all(np.isfinite(seg_toe)):
                toe_clear.append(float(seg_toe[0] - np.nanmin(seg_toe)))
        if swing_peaks == []: swing_peaks = None
        if toe_clear == []: toe_clear = None

    return {
        "side": label,
        "events": {
            "HS_idx": hs_idx, "TO_idx": to_idx, "MS_idx": ms_idx,
            "HS_ms": [int(t_ms[i]) for i in hs_idx],
            "TO_ms": [int(t_ms[i]) for i in to_idx],
            "MS_ms": [int(t_ms[i]) for i in ms_idx],
        },
        "metrics_knee_only": metrics_knee_only,
        "metrics_optional": {
            "swing_peak_flex_list": swing_peaks,
            "toe_clear_min_list": toe_clear,
            "stiff_knee_flag": bool(swing_peaks is not None and np.mean(swing_peaks) < GaitParams.stiff_knee_deg),
        },
    }

# ── gait: bilateral API
def detect_events_bilateral(npz_path: str, params: GaitParams | None = None) -> dict:
    params = params or GaitParams()
    D = load_npz(npz_path)
    lx, ly, t_ms = D["lm_x"], D["lm_y"], D["t_ms"]
    valid, meta = D["valid"], D["meta"]
    vmean, bbr, qok = D["vis_mean"], D["bbox_ratio"], D["quality_ok"]

    mask = valid.copy()
    if vmean is not None: mask &= (vmean >= params.vis_min)
    if bbr   is not None: mask &= (bbr >= params.bbox_min) & (bbr <= params.bbox_max)
    if qok   is not None: mask &= qok.astype(bool)
    lx = lx.copy(); ly = ly.copy()
    lx[~mask, :] = np.nan; ly[~mask, :] = np.nan

    hs_L, to_L = _hs_to_single_side(ly, t_ms, L_ANKLE, L_FOOT_INDEX, params.ma_win, params.min_step_interval_ms)
    hs_R, to_R = _hs_to_single_side(ly, t_ms, R_ANKLE, R_FOOT_INDEX, params.ma_win, params.min_step_interval_ms)

    pelvis_y = smooth1d((ly[:, L_HIP] + ly[:, R_HIP]) / 2.0, max(3, params.ma_win))
    ms_L = _ms_from_contra(pelvis_y, to_ctr=to_R, hs_ctr=hs_R) if (len(to_R) and len(hs_R)) else []
    ms_R = _ms_from_contra(pelvis_y, to_ctr=to_L, hs_ctr=hs_L) if (len(to_L) and len(hs_L)) else []

    yx = np.stack([ly, lx], axis=-1)
    knee_L = _knee_angle_deg(yx, L_HIP, L_KNEE, L_ANKLE)
    knee_R = _knee_angle_deg(yx, R_HIP, R_KNEE, R_ANKLE)

    left_res  = _side_result("LEFT",  t_ms, hs_L, to_L, ms_L, knee_L, ly, L_FOOT_INDEX, params)
    right_res = _side_result("RIGHT", t_ms, hs_R, to_R, ms_R, knee_R, ly, R_FOOT_INDEX, params)

    return {
        "task": "gait",
        "npz": str(npz_path),
        "meta": meta,
        "params": vars(params),
        "LEFT": left_res,
        "RIGHT": right_res,
    }

# ── sts: API
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

    # seat-off: 하강 속도 임계 통과 로컬 최소
    so_idx = []
    last_t = -1e9
    for i in range(1, len(v) - 1):
        if v[i] < v[i - 1] and v[i] < v[i + 1] and v[i] < params.v_pelvis_drop:
            if (t_ms[i] - last_t) > params.min_cycle_ms:
                so_idx.append(i); last_t = t_ms[i]

    # full-stand: 속도≈0, 무릎 안정
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

# ── save helpers
def save_events_json(result: dict, out_path: str | Path): _export_json(result, out_path)

def save_events_csv_timeline(result: dict, out_csv: str | Path):
    rows = []
    if result.get("task") == "gait":
        for side in ["LEFT", "RIGHT"]:
            ev = result.get(side, {}).get("events", {})
            for k in ["HS_ms", "TO_ms", "MS_ms"]:
                for t in ev.get(k, []):
                    rows.append({"side": side, "event": k, "time_ms": int(t)})
    else:
        ev = result.get("events", {})
        for k in ["seat_off_ms", "full_stand_ms"]:
            for t in ev.get(k, []):
                rows.append({"event": k, "time_ms": int(t)})
    _export_csv(rows, out_csv)

# ── CLI
def _print_summary_gait(res: dict):
    for side in ["LEFT", "RIGHT"]:
        ev = res.get(side, {}).get("events", {})
        ko = res.get(side, {}).get("metrics_knee_only", {})
        print(f"[{side}] HS n={len(ev.get('HS_ms', []))}, TO n={len(ev.get('TO_ms', []))}, MS n={len(ev.get('MS_ms', []))}")
        print(f"      Knee-only: max={ko.get('knee_max_deg',0):.1f}°, "
              f"hyperext_ratio(all)={ko.get('hyperext_ratio_all',0):.3f}, "
              f"longest={ko.get('hyperext_longest_ms',0)} ms")

def _print_summary_sts(res: dict):
    m = res.get("metrics", {})
    ev = res.get("events", {})
    print("STS:")
    print(f"  seat-off n={len(ev.get('seat_off_ms', []))}, full-stand n={len(ev.get('full_stand_ms', []))}")
    print(f"  cycles={m.get('cycles',0)}, mean_cycle_sec={m.get('mean_cycle_sec',0)}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Gait and STS events from pose npz")
    p.add_argument("--task", choices=["gait", "sts"], default="gait")
    p.add_argument("--npz", required=True)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-csv", action="store_true")
    a = p.parse_args()

    stem = Path(a.npz).stem
    if a.task == "gait":
        res = detect_events_bilateral(a.npz)
        _print_summary_gait(res)
        stem += "_bilateral"
    else:
        res = detect_sts_events(a.npz)
        _print_summary_sts(res)
        stem += "_sts"

    outdir = Path("results/reports"); outdir.mkdir(parents=True, exist_ok=True)
    if a.save_json:
        save_events_json(res, outdir / f"{stem}.json")
        print(f"[saved] results/reports/{stem}.json")
    if a.save_csv:
        save_events_csv_timeline(res, outdir / f"{stem}_timeline.csv")
        print(f"[saved] results/reports/{stem}_timeline.csv")
