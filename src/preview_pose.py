"""
파일명: src/preview_pose.py
 영상 파일을 Mediapipe Pose로 추정하여 스켈레톤을 그려
 미리보기 창으로 보여주고(---SHOW), 옵션으로 주석(annotated) 영상을 저장한다.
 회전(메타데이터/자동/수동)과 좌우반전, 창 닫힘 즉시 종료를 지원한다.

블록 구성
 0) 라이브러리 임포트: OpenCV, Mediapipe, argparse
 1) 유틸/초기화:
    - 회전 유틸, 메타데이터 기반 초기 회전값 추정
    - Pose 초기화(smooth_landmarks, conf 조정)
    - VideoWriter 준비(회전/반전 반영한 해상도로)
 2) 프레임 루프:
    - 루프 "시작"에서 창 닫힘 감지 → 즉시 종료
    - 프레임 읽기 → (회전/반전 적용) → Pose 추정 → 스켈레톤 그리기
    - FPS/가이드 텍스트 오버레이 → 미리보기/저장
    - 키 입력 처리: R(+90)/E(-90)/F(좌우반전)/Q 또는 ESC(종료)
 3) 종료/정리: 리소스 해제 및 창 파괴
 4) main/CLI: --video, --save-out, --resize, --complexity, --autorotate, --rotate, --flip

사용 방법
 1) 기본 미리보기:
    python src/preview_pose.py --video data/samples/sample_walk.mp4
 2) 세로 영상 자동 회전:
    python src/preview_pose.py --video data/samples/sample_walk.mp4 --autorotate
 3) 주석 영상 저장:
    python src/preview_pose.py --video data/samples/sample_walk.mp4 \
      --save-out results/figures/sample_walk_annot.mp4
 4) 실행 중 단축키:
    R(+90) / E(-90) / F(좌우반전) / Q 또는 ESC(종료) / 창 X(즉시 종료)

입력
 - mp4/mov/avi 등 일반 영상

출력
 - 미리보기 창(스켈레톤 오버레이)
 - (선택) 주석 영상 파일(--save-out 지정 시 mp4 저장)

참고(1차 출처)
 - Google Mediapipe Solutions Guide: https://ai.google.dev/edge/mediapipe/solutions/guide?hl=ko
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
import os
import time
import math
import argparse
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe가 필요합니다. 가상환경에서 `pip install mediapipe opencv-python` 실행하세요."
    ) from e


# 1) 유틸/초기화 ----------------------------------------------------------
def rotate_frame(img, deg: int):
    """deg ∈ {0,90,180,270} 회전"""
    d = int(deg) % 360
    if d == 0:   return img
    if d == 90:  return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180: return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270: return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img

def detect_orientation_meta(cap) -> int | None:
    """
    일부 OpenCV 빌드에 CAP_PROP_ORIENTATION_META 존재.
    값 예: 0/90/180/270. 없거나 인식 불가 시 None.
    """
    prop = getattr(cv2, "CAP_PROP_ORIENTATION_META", None)
    if prop is None:
        return None
    val = cap.get(prop)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if int(val) in (0, 90, 180, 270):
        return int(val)
    return None

def init_pose(model_complexity=1, det_conf=0.6, trk_conf=0.6):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        enable_segmentation=False,
        smooth_landmarks=True,              # ← 스무딩 활성화
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


# 2) 프리뷰 루틴 ----------------------------------------------------------
def preview(video_path, save_out=None, resize=None, model_complexity=1,
            autorotate=False, init_rotate=0, flip=False):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)

    # 초기 회전값 결정: 메타데이터 → autorotate(세로영상) → 수동 지정
    meta_rot = detect_orientation_meta(cap)
    if meta_rot is not None:
        rotate = meta_rot
    else:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        rotate = 90 if (autorotate and h > w) else int(init_rotate) % 360

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # Pose 초기화
    mp_pose, pose, drawer = init_pose(model_complexity=model_complexity)

    # 저장용 해상도 결정(회전/반전 반영)
    ok, tmp = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("첫 프레임을 읽지 못했습니다.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frm0 = rotate_frame(tmp, rotate)
    if flip:
        frm0 = cv2.flip(frm0, 1)
    H, W = frm0.shape[:2]
    writer = init_writer(save_out, W, H, fps)

    win_name = "Pose Preview (R:+90  E:-90  F:flip  Q/ESC:quit)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    last = time.time()
    try:
        while True:
            # 창 닫힘을 루프 "시작"에서 먼저 확인 → 재생성 문제 방지
            try:
                vis = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE)
                if vis < 1:
                    break
            except cv2.error:
                break

            ok, frame = cap.read()
            if not ok:
                break

            # (중요) 회전/반전 먼저 적용한 동일 프레임으로 추정
            frame_proc = rotate_frame(frame, rotate)
            if flip:
                frame_proc = cv2.flip(frame_proc, 1)

            rgb = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)
            rgb.flags.writeable = True

            # 랜드마크 그리기
            if res.pose_landmarks:
                drawer.draw_landmarks(frame_proc, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # FPS/가이드 오버레이
            now = time.time()
            fps_now = 1.0 / max(now - last, 1e-6)
            last = now
            cv2.putText(frame_proc, f"FPS: {fps_now:.1f}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_proc, "R:+90  E:-90  F:flip  Q/ESC:quit",
                        (12, frame_proc.shape[0]-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

            # 표시/저장
            display = cv2.resize(frame_proc, resize) if resize else frame_proc
            cv2.imshow(win_name, display)
            if writer is not None:
                writer.write(frame_proc)

            # 키 입력
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):           # q, ESC
                break
            elif key in (ord('r'), ord('R')):   # +90°
                rotate = (rotate + 90) % 360
            elif key in (ord('e'), ord('E')):   # -90°
                rotate = (rotate - 90) % 360
            elif key in (ord('f'), ord('F')):   # 좌우 반전 토글
                flip = not flip

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print("✅ 종료 완료")


# 3) main/CLI -------------------------------------------------------------
def _parse_resize(s: str | None):
    if not s:
        return None
    try:
        w, h = s.lower().split("x")
        return (int(w), int(h))
    except Exception:
        raise ValueError("--resize는 960x540 형식으로 입력하세요.")

def parse_args():
    p = argparse.ArgumentParser(description="Mediapipe Pose 영상 미리보기(회전/반전/저장/창닫힘)")
    p.add_argument("--video", required=True, help="입력 영상 경로 (예: data/samples/sample_walk.mp4)")
    p.add_argument("--save-out", default=None, help="주석(스켈레톤) 영상 저장 경로 (mp4)")
    p.add_argument("--resize", type=str, default=None, help="미리보기 창 크기 (예: 960x540)")
    p.add_argument("--complexity", type=int, default=1, choices=[0,1,2], help="Pose model_complexity")
    p.add_argument("--autorotate", action="store_true", help="세로(높이>너비) 영상은 자동으로 가로 회전")
    p.add_argument("--rotate", type=int, default=0, choices=[0,90,180,270], help="초기 회전 각도")
    p.add_argument("--flip", action="store_true", help="초기 좌우반전")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    preview(
        video_path=args.video,
        save_out=args.save_out,
        resize=_parse_resize(args.resize),
        model_complexity=args.complexity,
        autorotate=args.autorotate,
        init_rotate=args.rotate,
        flip=args.flip,
    )
