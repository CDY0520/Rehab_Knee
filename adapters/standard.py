"""
파일명: adapters/standard.py
 Mediapipe 기반 CSV를 표준 JSON으로 변환한다.
 (time_s, knee/hip/ankle 각도, 이벤트 검출 보조값 등 포함)

블록 구성
 0) 라이브러리 임포트: 표준/서드파티 모듈 로드
 1) 상수/유틸: 경로 유틸, 디렉토리 생성, float 변환
 2) CSV 파서: 헤더 자동 인식(필수/선택 컬럼) 후 레코드 리스트 생성
 3) 표준 JSON 구성: 프로젝트 공통 스키마로 time_series 배열 생성
 4) 저장 함수: results/json/ 경로에 파일 저장
 5) main/CLI: 인자 파싱(입력 CSV/출력 JSON/메타) 및 실행

사용 방법
 1) 가상환경 활성화 후 루트에서 실행: cd Rehab_Knee
  - python adapters/standard.py --csv data/samples/sample_walk_mediapipe.csv --out results/json/sample_gait.json --subject P001 --task gait --side right
 2) 출력 경로를 생략하면: results/json/<입력파일명_basename>.json 로 저장

입력
 - CSV (헤더 예시): time_s,knee_angle_deg,hip_angle_deg,ankle_angle_deg,toe_y,ankle_y

출력
 - JSON (예: results/json/sample_gait.json)
 - 스키마(요약): { "meta": {...}, "time_series": [ { "time_s": 0.000, "right_knee_angle": 175.2, ... }, ... ] }
 - 이벤트(events)는 이후 단계(events.py)에서 추가

출력 예시
 {
   "meta": {"subject_id":"P001","task":"gait","side":"right","source":"mediapipe"},
   "time_series": [
     {"time_s":0.000,"right_knee_angle":175.2,"right_hip_angle":160.3,"right_ankle_angle":95.7,"right_toe_y":0.823421,"right_ankle_y":0.761002},
     {"time_s":0.033,"right_knee_angle":174.9,"right_hip_angle":160.1,"right_ankle_angle":95.8,"right_toe_y":0.824100,"right_ankle_y":0.760800}
   ]
 }
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
import os
import csv
import json
import argparse

# 1) 상수/유틸 -----------------------------------------------------------
REQUIRED_COLS = ["time_s"]
OPTIONAL_MAP = {
    # CSV 컬럼명 → 표준 키
    "knee_angle_deg":  "knee_angle",
    "hip_angle_deg":   "hip_angle",
    "ankle_angle_deg": "ankle_angle",
    "toe_y":           "toe_y",
    "ankle_y":         "ankle_y",
}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def as_float_or_none(x: str):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None

def default_out_path(csv_path: str) -> str:
    base = os.path.splitext(os.path.basename(csv_path))[0]
    return os.path.join("results", "json", f"{base}.json")

# 2) CSV 파서 ------------------------------------------------------------
def parse_csv(csv_path: str):
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = [h.strip() for h in reader.fieldnames or []]

        # 필수 컬럼 체크
        for col in REQUIRED_COLS:
            if col not in headers:
                raise KeyError(f"입력 CSV에 '{col}' 컬럼이 없습니다. 현재 헤더: {headers}")

        rows = []
        for row in reader:
            rows.append({k.strip(): row.get(k, "").strip() for k in headers})
        return headers, rows

# 3) 표준 JSON 구성 ------------------------------------------------------
def build_standard_json(headers, rows, subject_id="UNKNOWN", task="gait", side="right", source="mediapipe"):
    """
    표준 스키마:
      meta: subject/task/side/source
      time_series: [{time_s, <side>_knee_angle, <side>_hip_angle, <side>_ankle_angle, <side>_toe_y, <side>_ankle_y}, ...]
    """
    side_prefix = "right" if side.lower() == "right" else "left"
    ts = []
    for r in rows:
        rec = {"time_s": as_float_or_none(r.get("time_s"))}
        for csv_key, std_key in OPTIONAL_MAP.items():
            val = as_float_or_none(r.get(csv_key))
            if val is not None:
                rec[f"{side_prefix}_{std_key}"] = val
        ts.append(rec)

    data = {
        "meta": {
            "subject_id": subject_id,
            "task": task,
            "side": side_prefix,
            "source": source,
        },
        "time_series": ts
    }
    return data

# 4) 저장 함수 -----------------------------------------------------------
def save_json(data, out_path: str):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 저장 완료: {out_path}")

# 5) main/CLI ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mediapipe CSV → 표준 JSON 변환기")
    p.add_argument("--csv", required=True, help="입력 CSV 경로")
    p.add_argument("--out", default=None, help="출력 JSON 경로(기본: results/json/<csv이름>.json)")
    p.add_argument("--subject", default="UNKNOWN", help="대상자 ID (예: P001)")
    p.add_argument("--task", default="gait", choices=["gait","sts","exercise","monitoring"], help="과제 타입")
    p.add_argument("--side", default="right", choices=["right","left"], help="측 선택")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    out_path = args.out or default_out_path(args.csv)
    headers, rows = parse_csv(args.csv)
    data = build_standard_json(headers, rows, subject_id=args.subject, task=args.task, side=args.side, source="mediapipe")
    save_json(data, out_path)
