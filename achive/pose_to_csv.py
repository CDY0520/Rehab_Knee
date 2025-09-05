"""
파일명: src/pose_to_csv.py
 일반 보행/STS 영상을 Mediapipe Pose로 자동 추정하여
 무릎/엉덩이/발목 각도(도)와 이벤트 검출용 보조값을 CSV로 저장

블록 구성
 0) 라이브러리 임포트: 표준/서드파티 모듈 로드
 1) 상수/유틸: 관절 인덱스 매핑, 경로 유틸, 디렉토리 생성
 2) 각도 계산 함수: 세 점(A-B-C)로 ∠ABC(도) 계산
 3) 한 프레임 처리: 랜드마크 → (무릎/엉덩이/발목 각도, 발끝/발목 y) 산출
 4) 비디오 전체 처리: 프레임 순회하며 시계열 CSV 기록
 5) main/CLI: 인자 파싱(영상 경로/출력 경로/측면) 및 실행

사용 방법
 1) 가상환경 활성화 후 루트에서 실행: cd Rehab_Knee
  - python src/pose_to_csv.py --video data/samples/sample_walk.mp4 --side right
 2) 왼쪽 다리를 기준으로 계산하고 싶으면: --side left

입력
 - mp4/mov/avi 등 일반 영상 (FPS 미표기 시 30fps로 가정)

출력
 - data/samples/sample_walk_mediapipe.csv)
 - 출력 CSV 컬럼: time_s, knee_angle_deg, hip_angle_deg, ankle_angle_deg, toe_y, ankle_y
 - time_s는 초 단위 실수, toe_y/ankle_y는 0~1 정규화 이미지 좌표의 y값
 - y값은 화면 위가 0, 아래가 1 (Mediapipe normalized)
 - 출력 예시: 0.000,5.21,14.98,0.82,0.76,0.71
"""

# 0) 라이브러리 임포트 -------------------------------------------------
import os
import csv
import math
import argparse
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe가 설치되어야 합니다. 가상환경에서 `pip install mediapipe opencv-python` 실행하세요."
    ) from e


# 1) 상수/유틸 ----------------------------------------------------------
# BlazePose 인덱스 (Right 기준; Left는 +1 오프셋 구조가 아닌 별도 인덱스)
POSE_IDX = {
    "right": {"shoulder": 12, "hip": 24, "knee": 26, "ankle": 28, "foot_index": 32},
    "left":  {"shoulder": 11, "hip": 23, "knee": 25, "ankle": 27, "foot_index": 31},
}

def ensure_dir(path: str):
    """출력 디렉토리 생성(없으면)"""
    os.makedirs(path, exist_ok=True)

def default_csv_path(video_path: str) -> str:
    """영상 경로로부터 기본 CSV 출력 경로 생성"""
    stem, _ = os.path.splitext(os.path.basename(video_path))
    return os.path.join("data", "samples", f"{stem}_mediapipe.csv")


# 2) 각도 계산 함수 ------------------------------------------------------
def angle_abc(a, b, c):
    """
    세 점(A-B-C)으로 꼭짓점 B의 내각(도)을 계산.
    a, b, c는 (x, y) 튜플. 값이 부족하면 None 반환.
    """
    if any(v is None for v in (a, b, c)):
        return None
    ax, ay = a
    bx, by = b
    cx, cy = c
    abx, aby = ax - bx, ay - by
    cbx, cby = cx - bx, cy - by
    nab = math.hypot(abx, aby)
    ncb = math.hypot(cbx, cby)
    if nab == 0 or ncb == 0:
        return None
    cosang = (abx * cbx + aby * cby) / (nab * ncb)
    cosang = max(min(cosang, 1.0), -1.0)
    return math.degrees(math.acos(cosang))


# 3) 한 프레임 처리 ------------------------------------------------------
def extract_angles_from_landmarks(lm, side="right"):
    """
    Mediapipe pose_landmarks에서 선택 측(right/left)의
    무릎/엉덩이/발목 각도와 toe_y/ankle_y를 계산해 반환.
    """
    idx = POSE_IDX[side]
    # normalized 좌표 (0~1); Mediapipe는 이미지 크기와 무관한 정규좌표를 제공
    def pt(i):
        return (lm[i].x, lm[i].y) if lm and i < len(lm) else None

    shoulder = pt(idx["shoulder"])
    hip      = pt(idx["hip"])
    knee     = pt(idx["knee"])
    ankle    = pt(idx["ankle"])
    foot     = pt(idx["foot_index"])

    knee_angle  = angle_abc(hip, knee, ankle)
    hip_angle   = angle_abc(shoulder, hip, knee)
    ankle_angle = angle_abc(knee, ankle, foot)

    toe_y   = foot[1]  if foot  else None
    ankle_y = ankle[1] if ankle else None

    return knee_angle, hip_angle, ankle_angle, toe_y, ankle_y


# 4) 비디오 전체 처리 ----------------------------------------------------
def process_video(video_path: str, out_csv: str = None, side: str = "right",
                  det_conf=0.5, trk_conf=0.5, model_complexity=1):
    """
    영상 전체를 순회하며 CSV 저장.
    - video_path: 입력 영상 경로
    - out_csv   : 출력 CSV 경로(미지정 시 video 파일명 기반 기본 경로)
    - side      : 'right' 또는 'left' (기본 right)
    """
    assert side in ("right", "left"), "side는 'right' 또는 'left'만 허용"

    if out_csv is None:
        out_csv = default_csv_path(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"비디오 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_idx = 0

    ensure_dir(os.path.dirname(out_csv))
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "time_s",
            "knee_angle_deg",
            "hip_angle_deg",
            "ankle_angle_deg",
            "toe_y",
            "ankle_y",
        ])

        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=det_conf,
            min_tracking_confidence=trk_conf,
        ) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(rgb)
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    knee_deg, hip_deg, ankle_deg, toe_y, ankle_y = extract_angles_from_landmarks(lm, side)
                else:
                    knee_deg = hip_deg = ankle_deg = toe_y = ankle_y = None

                time_s = frame_idx / fps
                writer.writerow([
                    f"{time_s:.3f}",
                    f"{knee_deg:.3f}"  if knee_deg  is not None else "",
                    f"{hip_deg:.3f}"   if hip_deg   is not None else "",
                    f"{ankle_deg:.3f}" if ankle_deg is not None else "",
                    f"{toe_y:.6f}"     if toe_y     is not None else "",
                    f"{ankle_y:.6f}"   if ankle_y   is not None else "",
                ])

                frame_idx += 1

    cap.release()
    print(f"✅ CSV 저장 완료: {out_csv}")


# 5) main/CLI ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mediapipe Pose로 관절 지표 CSV 생성")
    p.add_argument("--video", required=True, help="입력 영상 경로 (예: data/samples/sample_walk.mp4)")
    p.add_argument("--out",   default=None, help="출력 CSV 경로 (기본: <video>_mediapipe.csv)")
    p.add_argument("--side",  default="right", choices=["right", "left"], help="측 선택 (기본 right)")
    p.add_argument("--det",   type=float, default=0.5, help="min_detection_confidence")
    p.add_argument("--trk",   type=float, default=0.5, help="min_tracking_confidence")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path=args.video,
        out_csv=args.out,
        side=args.side,
        det_conf=args.det,
        trk_conf=args.trk,
    )
