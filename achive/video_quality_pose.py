"""
파일명: src/video_quality_pose.py
 Mediapipe Pose 랜드마크를 이용해 영상 품질을 정량화한다.
 (avg_visibility, visible_ratio, bbox_ratio를 계산하고 판정 리포트를 생성)

블록 구성
 0) 라이브러리 임포트
 1) 프레임 샘플링/전처리: 균등 샘플, RGB 변환
 2) Mediapipe 추정: 33 랜드마크 + visibility 수집
 3) 품질 지표 계산
    - avg_visibility: 프레임별 mean(visibility) → 전체 평균
    - visible_ratio : mean(visibility) >= vis_thr 인 프레임 비율
    - bbox_ratio    : (pose bbox 면적 / 프레임 면적) 평균
 4) 임계값 기반 판정: OK / WARN / FAIL
 5) main/CLI: 입력 경로/출력 경로/샘플 수 등 인자 파싱

사용 방법
 1) 가상환경 활성화 후 루트에서 실행: -cd Rehab_Knee
    - python src/video_quality_pose.py --video data/samples/sample_walk.mp4 --out results/json/sample_walk_pose_quality.json
 2) 옵션
    - --samples 120           : 샘플링 프레임 수
    - --vis-thr 0.5           : visible_ratio 판단 임계치
    - --bbox-min 0.12         : bbox_ratio 최소 권장 비율

입력
 - mp4/mov/avi 등 일반 영상

출력
 - JSON (예: results/json/sample_walk_pose_quality.json)
   {
     "avg_visibility": 0.78,
     "visible_ratio": 0.92,
     "bbox_ratio": 0.18,
     "verdict": "OK",
     "notes": ["가이드라인 충족", "가끔 랜드마크 불안정 프레임 존재(8%)"]
   }
"""

# 0) 라이브러리 임포트 ---------------------------------------------------
import os
import json
import argparse
import numpy as np
import cv2

try:
    import mediapipe as mp
except Exception as e:
    raise RuntimeError(
        "mediapipe가 필요합니다. (.venv)에서 `pip install mediapipe opencv-python` 실행"
    ) from e


# 1) 프레임 샘플링/전처리 -------------------------------------------------
def sample_indices(total_frames: int, n_samples: int) -> np.ndarray:
    """총 프레임 수에서 균등 간격으로 n개 인덱스 추출"""
    n = max(1, min(n_samples, max(1, total_frames)))
    return np.linspace(0, max(0, total_frames - 1), num=n, dtype=int)

def read_samples(video_path: str, n_samples: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = sample_indices(total, n_samples)
    frames_bgr = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames_bgr.append(f)
    cap.release()
    return frames_bgr


# 2) Mediapipe 추정 -------------------------------------------------------
def pose_infer_vis_bbox(frames_bgr):
    """
    각 프레임에서:
      - vis_mean: 33 랜드마크 visibility 평균(0~1)
      - bbox_ratio: 신체 bbox 면적 / 프레임 면적
    반환: vis_list, bbox_list
    """
    vis_list, bbox_list = [], []
    with mp.solutions.pose.Pose(static_image_mode=True) as pose:
        for f in frames_bgr:
            h, w = f.shape[:2]
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            res = pose.process(rgb)
            if not res.pose_landmarks:
                vis_list.append(0.0)
                bbox_list.append(0.0)
                continue

            lms = res.pose_landmarks.landmark
            vis = np.array([lm.visibility for lm in lms], dtype=float)
            xs = np.array([lm.x for lm in lms], dtype=float)
            ys = np.array([lm.y for lm in lms], dtype=float)

            # visibility 평균 (프레임 추적 안정성 척도)
            vis_list.append(float(np.clip(np.nanmean(vis), 0.0, 1.0)))

            # 바운딩 박스 면적 비율
            x1, x2 = np.clip([np.nanmin(xs), np.nanmax(xs)], 0, 1)
            y1, y2 = np.clip([np.nanmin(ys), np.nanmax(ys)], 0, 1)
            bw, bh = (x2 - x1) * w, (y2 - y1) * h
            bbox_ratio = (bw * bh) / (w * h + 1e-9)
            bbox_list.append(float(np.clip(bbox_ratio, 0.0, 1.0)))
    return vis_list, bbox_list


# 3) 품질 지표 계산 -------------------------------------------------------
def compute_quality_metrics(vis_list, bbox_list, vis_thr=0.5):
    """
    avg_visibility: mean(vis)
    visible_ratio : mean(vis) >= vis_thr 인 프레임 비율
    bbox_ratio    : mean(bbox_ratio)
    """
    vis_arr = np.array(vis_list, dtype=float)
    bbox_arr = np.array(bbox_list, dtype=float)

    avg_visibility = float(np.nanmean(vis_arr)) if vis_arr.size else 0.0
    visible_ratio = float(np.mean(vis_arr >= vis_thr)) if vis_arr.size else 0.0
    bbox_ratio = float(np.nanmean(bbox_arr)) if bbox_arr.size else 0.0
    return avg_visibility, visible_ratio, bbox_ratio


# 4) 임계값 기반 판정 -----------------------------------------------------
def verdict_from_metrics(avg_vis, vis_ratio, bbox_ratio, vis_ok=0.6, ratio_ok=0.8, bbox_min=0.12):
    """
    권장 임계:
      - avg_visibility >= 0.6
      - visible_ratio  >= 0.8  (80% 이상 유효 프레임)
      - bbox_ratio     >= 0.12 (피사체가 화면의 12% 이상)
    """
    notes = []
    bad = 0

    if avg_vis >= vis_ok: notes.append("avg_visibility 양호")
    else: notes.append(f"avg_visibility 낮음(<{vis_ok})"); bad += 1

    if vis_ratio >= ratio_ok: notes.append("visible_ratio 양호")
    else: notes.append(f"visible_ratio 낮음(<{ratio_ok})"); bad += 1

    if bbox_ratio >= bbox_min: notes.append("bbox_ratio 양호")
    else: notes.append(f"bbox_ratio 낮음(<{bbox_min})"); bad += 1

    verdict = "OK" if bad == 0 else ("WARN" if bad == 1 else "FAIL")
    return verdict, notes


# 5) main/CLI -------------------------------------------------------------
def analyze(video_path, out_path=None, samples=120, vis_thr=0.5, bbox_min=0.12):
    frames = read_samples(video_path, samples)
    vis_list, bbox_list = pose_infer_vis_bbox(frames)
    avg_vis, vis_ratio, bbox_ratio = compute_quality_metrics(vis_list, bbox_list, vis_thr=vis_thr)
    verdict, notes = verdict_from_metrics(avg_vis, vis_ratio, bbox_ratio,
                                          vis_ok=0.6, ratio_ok=0.8, bbox_min=bbox_min)

    report = {
        "avg_visibility": round(avg_vis, 3),
        "visible_ratio": round(vis_ratio, 3),
        "bbox_ratio": round(bbox_ratio, 3),
        "verdict": verdict,
        "notes": notes,
        "params": {"samples": samples, "vis_thr": vis_thr, "bbox_min": bbox_min}
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mediapipe Pose 기반 영상 품질 검사(avg_visibility/visible_ratio/bbox_ratio)")
    ap.add_argument("--video", required=True, help="입력 영상 경로")
    ap.add_argument("--out", default=None, help="JSON 리포트 경로")
    ap.add_argument("--samples", type=int, default=120, help="샘플링 프레임 수")
    ap.add_argument("--vis-thr", type=float, default=0.5, help="visible_ratio 판단 visibility 임계")
    ap.add_argument("--bbox-min", type=float, default=0.12, help="bbox_ratio 최소 권장 비율")
    args = ap.parse_args()
    analyze(args.video, args.out, samples=args.samples, vis_thr=args["vis-thr"] if hasattr(args, "__getitem__") else args.vis_thr, bbox_min=args.bbox_min)
