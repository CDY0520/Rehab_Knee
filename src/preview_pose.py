"""
파일명: src/preview_pose.py
 영상 파일을 Mediapipe Pose로 추정하여 프레임에 스켈레톤을 그려
 미리보기 창으로 보여주고(---SHOW), 옵션으로 주석 영상을 저장한다.

블록 구성
 0) 라이브러리 임포트: OpenCV, Mediapipe, argparse
 1) 초기화: Pose 모델/그리기 유틸, 비디오 캡처/출력 설정
 2) 프레임 루프: 추정 → 랜드마크 그리기 → FPS/가이드 오버레이 → 미리보기
 3) 종료/정리: 리소스 해제, 파일 저장 종료 메시지
 4) main/CLI: --video, --save-out, --resize, --complexity 등 인자 처리

사용 방법
 1) 가상환경 활성화 후 루트에서 실행: cd Rehab_Knee
  - python src/preview_pose.py --video data/samples/sample_walk.mp4
 2) 주석(스켈레톤) 영상 저장하고 싶으면:
  - python src/preview_pose.py --video data/samples/sample_walk.mp4 --save-out results/figures/sample_walk_annot.mp4
 3) 창에서 'q' 키를 누르면 종료

입력
 - mp4/mov/avi 등 일반 영상

출력
 - 미리보기 창(스켈레톤 오버레이)
 - (선택) 주석 영상 파일: --save-out 경로(mp4)

주의
 - Mediapipe 공식 가이드(https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ko)를 1차 출처로 따름
 - 창 포커스 상태에서 'q' 키 입력 시 종료
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
import os
import time
import argparse
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe가 필요합니다. (.venv)에서 `pip install mediapipe opencv-python` 실행"
    ) from e


# 1) 초기화 ---------------------------------------------------------------
def init_pose(model_complexity=1, det_conf=0.5, trk_conf=0.5):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        min_detection_confidence=det_conf,
        min_tracking_confidence=trk_conf,
    )
    drawer = mp.solutions.drawing_utils
    return mp_pose, pose, drawer


def init_writer(save_path, width, height, fps):
    if not save_path:
        return None
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(save_path, fourcc, fps if fps > 0 else 30.0, (width, height))


# 2) 프레임 루프 ----------------------------------------------------------
def preview(video_path, save_out=None, resize=None, model_complexity=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    mp_pose, pose, drawer = init_pose(model_complexity=model_complexity)
    writer = init_writer(save_out, width, height, fps)

    last = time.time()
    win_name = "Pose Preview (press 'q' to quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 추정 (BGR→RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)

        # 랜드마크 그리기
        if res.pose_landmarks:
            drawer.draw_landmarks(
                frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # FPS 오버레이
        now = time.time()
        fps_now = 1.0 / max(now - last, 1e-6)
        last = now
        cv2.putText(frame, f"FPS: {fps_now:.1f}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "press 'q' to quit", (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # 미리보기 창 크기 조정 옵션
        display = frame
        if resize:
            dw, dh = resize
            display = cv2.resize(frame, (dw, dh))

        cv2.imshow(win_name, display)

        # 저장(원본 해상도 기준)
        if writer is not None:
            writer.write(frame)

        # 종료 키(q)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"✅ 저장 완료: {save_out}")
    cv2.destroyAllWindows()


# 3) 종료/정리: (리소스 해제는 preview 내부에서 수행) -----------------------


# 4) main/CLI -------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Mediapipe Pose 영상 미리보기(---SHOW)")
    p.add_argument("--video", required=True, help="입력 영상 경로 (예: data/samples/sample_walk.mp4)")
    p.add_argument("--save-out", default=None, help="주석(스켈레톤) 영상 저장 경로 (예: results/figures/sample_walk_annot.mp4)")
    p.add_argument("--resize", type=str, default=None,
                   help="미리보기 창 크기 (예: 960x540). 원본 유지 시 생략")
    p.add_argument("--complexity", type=int, default=1, choices=[0,1,2],
                   help="Pose model_complexity (기본 1)")
    return p.parse_args()


def parse_resize(s):
    if not s:
        return None
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise ValueError("--resize 형식은 예: 960x540")


if __name__ == "__main__":
    args = parse_args()
    rs = parse_resize(args.resize)
    preview(
        video_path=args.video,
        save_out=args.save_out,
        resize=rs,
        model_complexity=args.complexity,
    )
