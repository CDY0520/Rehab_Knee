"""
파일명: run_analysis.py
기능: samples/ 아래 영상을 Mediapipe Pose로 처리하여
      - 보행(gait): HS/TO, stance·swing, 무릎/발목 지표 요약 → results/json/<video>_gait.json
      - STS: STS 시간(간이), ROM → results/json/<video>_sts.json
      - 운동(exercise): rep 카운트, ROM → results/json/<video>_exercise.json
      또한 (옵션) 주석(annotated) 영상을 results/figures/ 에 저장한다.

블록 구성
 0) 라이브러리 임포트
 1) 경로/환경: config 로드, 절대경로 유틸(resolve_path), 존재확인(must_exist)
 2) Mediapipe Pose 유틸 + 프레임→랜드마크 어댑터
 3) 공통 처리 (영상 읽기→frames 생성→주석 저장)
 4) 과제별 분석(gait / sts / exercise)
 5) 저장/요약 출력
 6) main/CLI

사용 예시
 1) 보행:
    python run_analysis.py --task gait --side LEFT --save-annot
 2) STS:
    python run_analysis.py --task sts --side LEFT --save-annot
 3) 운동:
    python run_analysis.py --task exercise --side LEFT --save-annot

참고
 - Mediapipe Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
import os, json, time, argparse
from typing import Dict, List, Tuple, Literal, Optional
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# 프로젝트 모듈/경로 설정 ---------------------------------------------------
try:
    # src/config.py가 있으면 우선 사용
    from src import config as CFG
    SAMPLES_DIR = getattr(CFG, "SAMPLES_DIR", "samples")
    RESULTS_JSON_DIR = getattr(CFG, "RESULTS_JSON_DIR", "results/json")
    RESULTS_FIG_DIR = getattr(CFG, "RESULTS_FIG_DIR", "results/figures")
except Exception:
    # 없으면 기본값
    SAMPLES_DIR = "samples"
    RESULTS_JSON_DIR = "results/json"
    RESULTS_FIG_DIR = "results/figures"

from src.metrics import (
    time_series_angle, series_xy, range_of_motion,
    summarize_knee_metrics, summarize_ankle_metrics
)
from src.events import (
    detect_heel_strike, detect_toe_off, stance_swing_masks, count_reps_from_angle
)

# 1) 경로/환경 유틸 --------------------------------------------------------
# 스크립트 기준 절대경로 변환
BASE_DIR = Path(__file__).resolve().parent

def resolve_path(p: str) -> str:
    """스크립트 위치(BASE_DIR) 기준 절대경로로 변환"""
    q = Path(p)
    if not q.is_absolute():
        q = (BASE_DIR / q).resolve()
    return str(q)

def must_exist(p: str, kind: str = "file"):
    """존재 확인 + 친절한 디버그 로그"""
    ok = os.path.isfile(p) if kind == "file" else os.path.isdir(p)
    if not ok:
        raise FileNotFoundError(
            f"[{kind.upper()} NOT FOUND] {p}\n"
            f"- CWD: {os.getcwd()}\n"
            f"- BASE_DIR: {BASE_DIR}\n"
            f"- Hint: 상대경로면 스크립트 기준으로 자동 변환되었습니다. 경로/파일명을 확인하세요."
        )

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def default_out_paths(video_path: str, task: str) -> Tuple[str, str]:
    """
    입력 영상 경로 → 결과 경로 제안
    - json:   results/json/<stem>_<task>.json
    - annot:  results/figures/<stem>_<task>_annot.mp4
    (두 경로 모두 BASE_DIR 기준 절대경로로 변환)
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    json_path = resolve_path(os.path.join(RESULTS_JSON_DIR, f"{stem}_{task}.json"))
    annot_path = resolve_path(os.path.join(RESULTS_FIG_DIR,  f"{stem}_{task}_annot.mp4"))
    return json_path, annot_path

# 2) Mediapipe Pose 유틸 ---------------------------------------------------
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
        h, w = frame.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), -rotate_deg, 1.0)
        return cv2.warpAffine(frame, M, (w, h))

def draw_text(img, text, xy=(10, 22), scale=0.6, color=(255,255,255)):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

def mp_landmarks_to_dict(landmarks, image_w, image_h) -> Dict[str, Tuple[float,float,float,float]]:
    """
    Mediapipe NormalizedLandmark 리스트 → 프로젝트 공통 dict로 변환
    (정규화 0~1 좌표로 통일)
    """
    name_map = mp_pose.PoseLandmark
    lm = {}
    for idx, l in enumerate(landmarks):
        name = name_map(idx).name
        lm[name] = (float(l.x), float(l.y), float(l.z), float(l.visibility))
    return lm

# 3) 공통 처리 (영상→frames) ----------------------------------------------
def process_video_to_frames(video_path: str,
                            resize: Optional[int],
                            rotate: int,
                            flip: bool,
                            complexity: int,
                            save_annot: bool,
                            annot_out_path: Optional[str]) -> Tuple[List[Dict], float]:
    """
    영상→프레임 루프→Pose 추정→frames(list[dict]) 생성.
    (save_annot=True면 주석 영상을 annot_out_path에 저장)
    반환: (frames, fps)
    """
    abs_video_path = resolve_path(video_path)
    must_exist(abs_video_path, "file")
    print(f"[INFO] Opening video: {abs_video_path}")

    # FFmpeg 백엔드 우선 시도 → 실패 시 일반 백엔드 재시도
    cap = cv2.VideoCapture(abs_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(abs_video_path)
    if not cap.isOpened():
        # 디버그: 확장자/파일크기/빌드정보
        try:
            size = os.path.getsize(abs_video_path)
        except Exception:
            size = -1
        raise RuntimeError(
            "Cannot open video: {p}\n"
            " - size(bytes): {sz}\n"
            " - OpenCV build info(FFmpeg 포함 여부):\n{bi}".format(
                p=abs_video_path, sz=size, bi=cv2.getBuildInformation()[:800]
            )
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = None
    if save_annot:
        ensure_dir(annot_out_path)

    frames: List[Dict] = []
    idx = 0
    t0 = time.time()
    last_lm = None # 마지막으로 성공한 랜드마크를 저장해서 forward-fill

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=int(complexity),
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

            # 랜드마크 dict 변환 (+forward-fill)
            if result.pose_landmarks:
                h, w = frame.shape[:2]
                lm_dict = mp_landmarks_to_dict(result.pose_landmarks.landmark, w, h)
                last_lm = lm_dict
                frames.append({"index": idx, "landmarks": lm_dict})
            else:
                # 첫 유효 프레임이 나오기 전에는 건너뜀, 이후엔 마지막 값으로 유지
                if last_lm is not None:
                    frames.append({"index": idx, "landmarks": last_lm})
                # else: 아직 없는 경우는 스킵 (주석영상만 저장)

            # 주석 저장
            if save_annot:
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(annot_out_path, fourcc, fps, (frame.shape[1], frame.shape[0]))
                annot = frame.copy()
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        annot, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,128,255), thickness=2)
                    )
                disp_fps = (idx+1) / max(1e-6, (time.time()-t0))
                draw_text(annot, f"{os.path.basename(abs_video_path)}  ~{disp_fps:4.1f} FPS", (10, 22))
                writer.write(annot)

            idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    return frames, float(fps)

# 4) 과제별 분석 ------------------------------------------------------------
def analyze_gait(frames: List[Dict], fps: float, side: Literal["LEFT","RIGHT"]) -> Dict:
    heel_y = [1.0 - f["landmarks"].get(f"{side}_HEEL", (0, 0, 0, 0))[1] for f in frames]
    toe_y = [1.0 - f["landmarks"].get(f"{side}_FOOT_INDEX", (0, 0, 0, 0))[1] for f in frames]

    hs = detect_heel_strike(heel_y, fps)
    to = detect_toe_off(toe_y, fps)
    stance_mask, swing_mask = stance_swing_masks(len(frames), hs, to)

    knee_sum  = summarize_knee_metrics(frames, side, stance_mask, swing_mask)
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
    knee_deg = time_series_angle(frames, f"{side}_HIP", f"{side}_KNEE", f"{side}_ANKLE")
    rep_count, rep_starts, rep_ends = count_reps_from_angle(knee_deg, fps, min_rep_sec=0.6, prominence=5)
    return {
        "task": "exercise",
        "side": side,
        "events": {"rep_starts": rep_starts, "rep_ends": rep_ends},
        "metrics": {"rep_count": rep_count, "knee_rom_deg": range_of_motion(knee_deg)}
    }

# 5) 저장/요약 --------------------------------------------------------------
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
        print(f"Knee ROM: {knee['knee_rom_deg']:.2f} deg | Hyperext: {int(knee['knee_hyperextension_flag'])} | Swing peak flex: {knee['knee_swing_peak_flex_deg']:.2f} deg")
        print(f"Ankle ROM: {ankle['ankle_rom_deg']:.2f} deg | Dorsi max: {ankle['dorsi_max_deg']:.2f} | Plantar max: {ankle['plantar_max_deg']:.2f}")
    elif task == "sts":
        print(f"STS time: {result['metrics']['sts_time_sec']:.2f} s | Knee ROM: {result['metrics']['knee_rom_deg']:.2f} deg")
    else:
        print(f"Rep count: {result['metrics']['rep_count']} | Knee ROM: {result['metrics']['knee_rom_deg']:.2f} deg")
    print("=====================================\n")

# 6) main/CLI --------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gait","sts","exercise"], help="분석 유형")
    ap.add_argument("--video", default=os.path.join(SAMPLES_DIR, "sample_walk.mp4"), help="입력 영상 경로")
    ap.add_argument("--side", default="LEFT", choices=["LEFT","RIGHT"], help="좌/우 분석 기준")
    ap.add_argument("--rotate", type=int, default=0, help="회전 각도(+90/-90/180 등)")
    ap.add_argument("--flip", action="store_true", help="좌우반전")
    ap.add_argument("--resize", type=int, default=None, help="긴 변 기준 리사이즈(px)")
    ap.add_argument("--complexity", type=int, default=1, help="Mediapipe model_complexity(0/1/2)")
    ap.add_argument("--save-annot", action="store_true", help="주석 영상 저장 여부")
    ap.add_argument("--out-json", default=None, help="결과 JSON 저장 경로(미지정 시 기본 경로 사용)")
    ap.add_argument("--out-annot", default=None, help="주석 영상 저장 경로(미지정 시 기본 경로 사용)")
    args = ap.parse_args()

    # 출력 경로 기본값 생성(절대경로)
    json_path, annot_path = default_out_paths(args.video, args.task)
    if args.out_json:  json_path = resolve_path(args.out_json)
    if args.out_annot: annot_path = resolve_path(args.out_annot)

    # 처리
    video_abs = resolve_path(args.video)
    frames, fps = process_video_to_frames(
        video_path=video_abs,
        resize=args.resize,
        rotate=args.rotate,
        flip=args.flip,
        complexity=args.complexity,
        save_annot=args.save_annot,
        annot_out_path=annot_path
    )

    # 분석
    if args.task == "gait":
        result = analyze_gait(frames, fps, args.side)
    elif args.task == "sts":
        result = analyze_sts(frames, fps, args.side)
    else:
        result = analyze_exercise(frames, fps, args.side)

    # 저장/요약
    save_json(result, json_path)
    print_summary(result)
    print(f"[저장 완료] JSON: {json_path}")
    if args.save_annot:
        print(f"[저장 완료] Annotated MP4: {annot_path}")

if __name__ == "__main__":
    main()
