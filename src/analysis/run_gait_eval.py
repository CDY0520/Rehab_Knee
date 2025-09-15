"""
파일명: src/analysis/run_gait_eval.py

설명:
  - Mediapipe Pose npz 입력을 events.py API로 분석한다.
  - 보행(Gait) 이벤트/지표를 검출하고 JSON과 평가용 CSV를 생성한다.
  - 출력:
      · results/experioment/{video_id}_gait.json
      · results/experioment/pred_{video_id}.csv (video_id,side,event,time_ms)

사용법 예시:
  python src/analysis/run_gait_eval.py results/keypoints/sample_walk_normal.npz normal
  python src/analysis/run_gait_eval.py results/keypoints/sample_walk.npz hyperext

블록 구성:
  0) import 및 경로 설정
  1) events.py API 호출 (detect_events_bilateral)
  2) JSON 저장 함수
  3) Pred CSV 변환 함수 (HS/TO/MS/GENU_RECURVATUM)
  4) 실행 함수(run) → JSON/CSV 출력
  5) CLI 엔트리포인트
"""

import sys, json, csv
from pathlib import Path

# 프로젝트 루트 추가(../.. 에 events.py 있음)
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from events import detect_events_bilateral  # 네가 제공한 events.py

OUT_DIR = Path("results/experiment")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_pred_csv(result: dict, out_csv: Path, video_id: str):
    rows = []
    for side in ["LEFT", "RIGHT"]:
        ev = result.get(side, {}).get("events", {})
        # HS/TO/MS/GENU_RECURVATUM을 time_ms 기준으로 납작한 행으로 변환
        for k in ["HS_ms", "TO_ms", "MS_ms", "GENU_RECURVATUM_ms"]:
            for t in ev.get(k, []):
                # 이벤트명 표준화: HS_ms -> HS, TO_ms -> TO, ...
                name = k.replace("_ms", "")
                rows.append({"video_id": video_id, "side": side[0], "event": name, "time_ms": int(t)})
    rows.sort(key=lambda r: (r["side"], r["time_ms"]))

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_id","side","event","time_ms"])
        w.writeheader(); w.writerows(rows)

def run(npz_path: str, video_id: str):
    res = detect_events_bilateral(npz_path)
    save_json(res, OUT_DIR / f"{video_id}_gait.json")
    write_pred_csv(res, OUT_DIR / f"pred_{video_id}.csv", video_id)
    print(f"[saved] JSON  : results/experiment/{video_id}_gait.json")
    print(f"[saved] Pred  : results/experiment/pred_{video_id}.csv")

if __name__ == "__main__":
    # 예시:
    # python src/analysis/run_gait_eval.py results/keypoints/sample_walk_normal.npz normal
    # python src/analysis/run_gait_eval.py results/keypoints/sample_walk.npz hyperext
    npz = sys.argv[1]
    vid = sys.argv[2]
    run(npz, vid)
