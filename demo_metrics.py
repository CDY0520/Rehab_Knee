"""
파일명: demo_metrics.py
기능: 샘플 영상(mp4 등)을 Mediapipe Pose로 처리하여
      - 보행(HS/TO, stance·swing, 무릎/발목 지표)
      - STS(시작~종료 시간)
      - 운동(rep 카운트)
      를 계산하고, 주석(annotated) 영상을 저장하며, 지표를 JSON으로 저장한다.

블록 구성
 0) 라이브러리 임포트
 1) 유틸/초기화:
    - Mediapipe Pose 초기화
    - 프레임→랜드마크 dict 변환 어댑터
    - 오버레이 드로잉, 텍스트 유틸
 2) 공통 처리 함수:
    - 영상 읽어 프레임 루프 (옵션 회전/좌우반전/리사이즈)
    - 프레임별 포즈 추정 → frames(list[dict]) 생성
    - (선택) 주석 프레임 작성 및 VideoWriter 저장
 3) 과제(task)별 분석:
    - gait: HS/TO → stance/swing → 무릎/발목 지표 산출
    - sts : STS 시간(간이; rep 로직 응용) 산출
    - exercise: 관절각 기반 rep 카운트
 4) 저장/리포트:
    - metrics/events 결과를 JSON으로 저장
    - 콘솔에 핵심치 요약 출력
 5) main/CLI:
    --video, --task [gait|sts|exercise], --side, --save-annot, --save-json,
    --rotate, --flip, --resize, --complexity

사용 방법 (예시)
 1) 보행:
    python demo_metrics.py --video data/samples/gait.mp4 --task gait --side LEFT \
        --save-annot results/gait_annot.mp4 --save-json results/gait_result.json
 2) STS:
    python demo_metrics.py --video data/samples/sts.mp4 --task sts --side LEFT \
        --save-annot results/sts_annot.mp4 --save-json results/sts_result.json
 3) 운동(스쿼트/SLR 등):
    python demo_metrics.py --video data/samples/exercise.mp4 --task exercise --side LEFT \
        --save-annot results/ex_annot.mp4 --save-json results/ex_result.json

입력
 - mp4/mov/avi 등 영상 파일

출력
 - (선택) 주석 영상 파일(--save-annot 지정 시)
 - (선택) 결과 JSON (--save-json 지정 시; 지표/이벤트 포함)

참고(1차 출처)
 - Mediapipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
import os, json, time, argparse
from typing import Dict, List, Tuple, Literal, Optional

import cv2
import numpy as np
import mediapipe as mp

from src.metrics import (
    time_series_angle, series_xy, foot_axis_angle_series, range_of_motion,
    summarize_knee_metrics, summarize_ankle_metrics
)
from src.events import (
    detect_heel_strike, detect_toe_off, stance_swing_masks, count_reps_from_angle
)

# 1) 유틸/초기화 ----------------------------------------------------------
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def rotate_frame(frame, rotate_deg: int):
    """+90=시계방향 90°, -90, 180 지원"""
    if rotate_deg % 360 == 0:
        return frame
    rot = rotate_deg % 360
    if rot == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rot == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        # 임의 각도는 getRotationMatrix2D로 처리
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), -rotate_deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h))

def draw_text(img, text, xy=(10, 30), scale=0.6, color=(255,255,255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def mp_landmarks_to_dict(landmarks, image_w, image_h) -> Dict[str, Tuple[float,float,float,float]]:
    """
    Mediapipe PoseWorld/Normalized landmarks → 프로젝트 공통 dict로 변환
    (정규화 0~1 좌표로 통일: x/w, y/h)
    """
    name_map = mp_pose.PoseLandmark
    lm = {}
    for idx, l in enumerate(landmarks):
        name = name_map(idx).name  # 'LEFT_HIP' 등
        lm[name] = (float(l.x), float(l.y), float(l.z), float(l.visibility))
    return lm

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

# 2) 공통 처리 함수 --------------------------------------------------------
def process_video_to_frames(video_path: str,
                            resize: Optional[int],
                            rotate: int,
                            flip: bool,
                            complexity: int,
                            save_annot_path: Optional[str]=None) -> Tuple[List[Dict], float]:
    """
    영상→프레임 루프→Pose 추정→frames(list[dict]) 생성.
    (선택) 주석 프레임을 VideoWriter로 저장.
    반환: (frames, fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter 준비
    writer = None
    if save_annot_path:
        ensure_dir(save_annot_path)

    frames: List[Dict] = []
    idx = 0
    t0 = time.time()

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=complexity,
        enable_segmentation=False,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 회전/반전/리사이즈
            frame = rotate_frame(frame, rotate)
            if flip:
                frame = cv2.flip(frame, 1)
            if resize:
                h, w = frame.shape[:2]
                scale = resize / max(h, w)
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))

            # Pose 추정
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            # 랜드마크 dict 변환
            if result.pose_landmarks:
                h, w = frame.shape[:2]
                lm_dict = mp_landmarks_to_dict(result.pose_landmarks.landmark, w, h)
                frames.append({"index": idx, "landmarks": lm_dict})
            else:
                frames.append({"index": idx, "landmarks": {}})

            # 주석 영상
            if save_annot_path:
                # writer 초기화(첫 프레임 때)
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(save_annot_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

                annot = frame.copy()
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annot, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,128,255), thickness=2)
                    )
                # FPS overlay
                elapsed = time.time() - t0
                disp_fps = (idx+1)/elapsed if elapsed > 0 else 0.0
                draw_text(annot, f"preview ~{disp_fps:4.1f} FPS", (10, 24))
                writer.write(annot)

            idx += 1

    cap.release()
    if writer is not None:
        writer.release()

    return frames, float(fps)

# 3) 과제(task)별 분석 ------------------------------------------------------
def analyze_gait(frames: List[Dict], fps: float, side: Literal["LEFT","RIGHT"]) -> Dict:
    """보행 분석: HS/TO → stance/swing → 무릎/발목 지표 요약"""
    heel_y = [f["landmarks"].get(f"{side}_HEEL", (0,0,0,0))[1] for f in frames]
    toe_y  = [f["landmarks"].get(f"{side}_FOOT_INDEX", (0,0,0,0))[1] for f in frames]

    hs = detect_heel_strike(heel_y, fps)
    to = detect_toe_off(toe_y, fps)
    stance_mask, swing_mask = stance_swing_masks(len(frames), hs, to)

    knee_sum = summarize_knee_metrics(frames, side, stance_mask, swing_mask)
    ankle_sum = summarize_ankle_metrics(frames, side, stance_mask, swing_mask, scale_cm_per_unit=None)

    return {
        "task": "gait",
        "side": side,
        "events": {"heel_strike": hs, "toe_off": to},
        "masks": {"stance": stance_mask, "swing": swing_mask},
        "knee_metrics": knee_sum,
        "ankle_metrics": ankle_sum
    }

def analyze_sts(frames: List[Dict], fps: float, side: Literal["LEFT","RIGHT"]) -> Dict:
    """STS 분석(간이): 무릎각 시계열 기반으로 valley→peak→valley 1회로 보고 전체 시간 산출"""
    knee_deg = time_series_angle(frames, f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE")
    rep_count, rep_starts, rep_ends = count_reps_from_angle(knee_deg, fps, min_rep_sec=0.8, prominence=6)
    duration_sec = 0.0
    if rep_count >= 1:
        duration_sec = max(0.0, (rep_ends[0] - rep_starts[0]) / fps)

    return {
        "task": "sts",
        "side": side,
        "events": {"rep_starts": rep_starts, "rep_ends": rep_ends},
        "metrics": {"sts_time_sec": duration_sec, "knee_rom_deg": range_of_motion(knee_deg)}
    }

def analyze_exercise(frames: List[Dict], fps: float, side: Literal["LEFT","RIGHT"]) -> Dict:
    """운동(스쿼트/SLR 등) 분석: 무릎각 시계열 peak/valley 기반 rep 카운트"""
    knee_deg = time_series_angle(frames, f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE")
    rep_count, rep_starts, rep_ends = count_reps_from_angle(knee_deg, fps, min_rep_sec=0.6, prominence=5)

    return {
        "task": "exercise",
        "side": side,
        "events": {"rep_starts": rep_starts, "rep_ends": rep_ends},
        "metrics": {
            "rep_count": rep_count,
            "knee_rom_deg": range_of_motion(knee_deg)
        }
    }

# 4) 저장/리포트 ------------------------------------------------------------
def save_json(obj, path: str):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def print_summary(result: Dict):
    task = result["task"]
    print(f"\n=== SUMMARY ({task}) ===")
    if task == "gait":
        hs = len(result["events"]["heel_strike"])
        to = len(result["events"]["toe_off"])
        knee = result["knee_metrics"]; ankle = result["ankle_metrics"]
        print(f"HS n={hs}, TO n={to}")
        print(f"Knee ROM: {knee['knee_rom_deg']:.2f} deg, Hyperext flag: {int(knee['knee_hyperextension_flag'])}")
        print(f"Knee swing peak flex: {knee['knee_swing_peak_flex_deg']:.2f} deg")
        print(f"Ankle ROM: {ankle['ankle_rom_deg']:.2f} deg, Dorsi max: {ankle['dorsi_max_deg']:.2f} deg, Plantar max: {ankle['plantar_max_deg']:.2f} deg")
    elif task == "sts":
        print(f"STS time: {result['metrics']['sts_time_sec']:.2f} s, Knee ROM: {result['metrics']['knee_rom_deg']:.2f} deg")
    else:
        print(f"Rep count: {result['metrics']['rep_count']}, Knee ROM: {result['metrics']['knee_rom_deg']:.2f} deg")
    print("==============\n")

# 5) main/CLI --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="입력 영상 경로")
    ap.add_argument("--task", required=True, choices=["gait","sts","exercise"], help="분석 유형")
    ap.add_argument("--side", default="LEFT", choices=["LEFT","RIGHT"], help="분석 기준 측")
    ap.add_argument("--save-annot", default=None, help="주석 영상 mp4 저장 경로")
    ap.add_argument("--save-json", default=None, help="결과 JSON 저장 경로")
    ap.add_argument("--rotate", type=int, default=0, help="회전 각도(+90/-90/180 등)")
    ap.add_argument("--flip", action="store_true", help="좌우반전")
    ap.add_argument("--resize", type=int, default=None, help="긴 변을 이 값으로 리사이즈(픽셀)")
    ap.add_argument("--complexity", type=int, default=1, help="Mediapipe Pose model_complexity (0/1/2)")
    args = ap.parse_args()

    # 1) 영상→frames 변환(+주석 저장)
    frames, fps = process_video_to_frames(
        video_path=args.video,
        resize=args.resize,
        rotate=args.rotate,
        flip=args.flip,
        complexity=args.complexity,
        save_annot_path=args.save_annot
    )

    # 2) 과제별 분석 실행
    if args.task == "gait":
        result = analyze_gait(frames, fps, args.side)
    elif args.task == "sts":
        result = analyze_sts(frames, fps, args.side)
    else:
        result = analyze_exercise(frames, fps, args.side)

    # 3) 저장/요약
    if args.save_json:
        save_json(result, args.save_json)
    print_summary(result)

if __name__ == "__main__":
    main()
