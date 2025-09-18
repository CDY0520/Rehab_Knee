# -*- coding: utf-8 -*-
"""
파일명: src/analysis/run_gait_eval.py

설명:
  - Mediapipe Pose npz 입력을 events.py API(detect_events_bilateral)로 분석.
  - Gait 이벤트를 JSON/CSV로 저장.
  - CSV는 기본 HS/TO/MS만 저장(=GT와 직접 비교용). --include-gr로 GR 추가 가능.

사용 예:
  python src/analysis/run_gait_eval.py results/keypoints/sample_walk_normal.npz normal
  python src/analysis/run_gait_eval.py results/keypoints/sample_walk.npz hyperext --include-gr
"""

import sys, json, csv, argparse
import numpy as np
from pathlib import Path


# 프로젝트 루트 추가(../.. 에 events.py 있음)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from events import detect_events_bilateral  # src/events.py

OUT_DIR = Path("results/experiment")  # 오타 수정
OUT_DIR.mkdir(parents=True, exist_ok=True)

def _convert_json_safe(obj):
    """numpy 타입을 json 직렬화 가능한 파이썬 타입으로 변환"""
    if isinstance(obj, (np.integer, )):
        return int(obj)
    elif isinstance(obj, (np.floating, )):
        return float(obj)
    elif isinstance(obj, (np.ndarray, )):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_json_safe(v) for v in obj]
    else:
        return obj

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj_safe = _convert_json_safe(obj)   # numpy 타입 → 파이썬 타입 변환
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj_safe, f, ensure_ascii=False, indent=2)

def write_pred_csv(result: dict, out_csv: Path, video_id: str,
                   include_gr: bool = False, side_keep: str | None = None):
    """
    HS/TO/MS는 events.HS_ms/TO_ms/MS_ms 사용.
    GENU_RECURVATUM은 cycles_metrics에서 HYPEREXT_LEVEL!='none'인 사이클의 MS 시점(ms_ms)을 사용.
    side_keep: 'L' 또는 'R'이면 해당 사이드만 저장.
    """
    rows = []
    keep_ev = ["HS_ms", "TO_ms", "MS_ms"]

    for side_key in ["LEFT", "RIGHT"]:
        side_char = side_key[0]  # 'L' or 'R'
        if side_keep and side_char != side_keep.upper():
            continue

        ev = result.get(side_key, {}).get("events", {})
        # HS/TO/MS
        for k in keep_ev:
            for t in ev.get(k, []):
                rows.append({
                    "video_id": video_id,
                    "side": side_char,
                    "event": k.replace("_ms", ""),   # HS / TO / MS
                    "time_ms": int(t)
                })

        # GENU_RECURVATUM from cycle-level decision
        if include_gr:
            cycles = result.get(side_key, {}).get("cycles_metrics", []) or []
            for r in cycles:
                if r.get("HYPEREXT_LEVEL") and r["HYPEREXT_LEVEL"] != "none":
                    ms_t = int(r.get("ms_ms", r.get("ms_ms".upper(), 0)))
                    rows.append({
                        "video_id": video_id,
                        "side": side_char,
                        "event": "GENU_RECURVATUM",
                        "time_ms": ms_t
                    })

    rows.sort(key=lambda r: r["time_ms"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "side", "event", "time_ms"])
        w.writeheader()
        w.writerows(rows)


def run(npz_path: str, video_id: str, include_gr: bool = False):
    res = detect_events_bilateral(npz_path)
    save_json(res, OUT_DIR / f"{video_id}_gait.json")
    write_pred_csv(res, OUT_DIR / f"pred_{video_id}.csv", video_id, include_gr)
    print(f"[saved] JSON  : {OUT_DIR}/{video_id}_gait.json")
    print(f"[saved] Pred  : {OUT_DIR}/pred_{video_id}.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("video_id")
    ap.add_argument("--include-gr", action="store_true", help="CSV에 GENU_RECURVATUM 포함")
    args = ap.parse_args()
    run(args.npz, args.video_id, args.include_gr)
