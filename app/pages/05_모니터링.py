"""
파일명: app/pages/05_스쿼트_모니터링.py

기능: 업로드된 스쿼트 영상을 분석하여 반복수, 깊이, 템포, 좌우 정렬 이상(무릎 안짖음, 힙윙크 등)을 모니터링하고 리포트 제공

사용 라이브러리: streamlit, opencv-python, numpy, pandas, mediapipe

입력: 사용자가 업로드한 스쿼트 영상(mp4, mov 등)

출력: 반복 검출 결과 테이블, 경고 지표, CSV 다운로드 파일

블록 구성:
   0) UI 기본 설정
   1) 파라미터 패널 (임계값, 경고 기준 등)
   2) 유틸 함수 (각도, 좌표, 필터링 등)
   3) 업로드 처리
   4) 프레임 루프 → 포즈 추정 및 시계열 데이터 수집
   5) 후처리 및 반복 검출 FSM
   6) 리포트 테이블 생성
   7) 요약 및 결과 표시
   8) 주의사항 안내
"""

import os
import io
import math
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    import mediapipe as mp
except Exception as e:
    st.error("mediapipe 불러오기 실패. requirements 확인.")
    st.stop()

# -----------------------------
# 0) UI 기본 설정
# -----------------------------
st.set_page_config(page_title="스쿼트 모니터링", page_icon="🏋️", layout="wide")
st.title("🏋️ 스쿼트 모니터링")

# -----------------------------
# 1) 파라미터 패널
# -----------------------------
with st.sidebar:
    st.header("설정")
    view = st.selectbox("촬영 시야", ["측면", "정면"], index=0)
    side_pref = st.selectbox("기준 측", ["LEFT", "RIGHT"], index=0)
    knee_flex_start = st.slider("반복 시작 기준(무릎 굴곡, 도)", 40, 120, 70)  # 하강시
    knee_flex_bottom = st.slider("바닥 지점(최대 굴곡 최소치, 도)", 60, 150, 100)
    knee_extend_end = st.slider("반복 종료 기준(무릎 신전, 도)", 150, 179, 165)     # 상승 완료
    depth_rule = st.selectbox("깊이 판정 규칙", ["힙이 무릎 아래", "무릎각 기준"], index=0)
    hip_below_knee_margin = st.slider("힙<무릎 마진(px)", 0, 40, 10)
    torso_lean_warn = st.slider("상체 전경사 경고(도)", 0, 80, 55)
    wink_thresh_deg = st.slider("힙 윙크(골반 후방경사) 경고(도)", 0, 30, 12)
    valgus_ratio_warn = st.slider("무릎 안짖음 경고(정면) 비율", 0.0, 0.5, 0.15, 0.01)
    min_rep_duration = st.slider("최소 반복 시간(초)", 0.2, 5.0, 1.0, 0.1)
    smooth_win = st.slider("각도 평활 이동평균(프레임)", 1, 9, 5, 2)
    show_annot = st.checkbox("프레임에 주석 표시", True)

# -----------------------------
# 2) 유틸 함수
# -----------------------------
def angle(a, b, c):
    """세 점(a,b,c)의 내각 b(도). 각 점은 (x,y) ndarray."""
    ba, bc = a - b, c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def vec_angle_with_vertical(p_top, p_bottom):
    """세로축 대비 기울기(도). 0도=수직. 값이 클수록 전경사."""
    v = p_bottom - p_top
    vertical = np.array([0.0, 1.0])
    cosang = np.dot(v, vertical) / (np.linalg.norm(v) * np.linalg.norm(vertical) + 1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def moving_avg(x, k):
    if k <= 1:
        return x
    s = pd.Series(x).rolling(k, min_periods=1, center=True).mean()
    return s.values

def select_side(lmk, side_pref):
    """가시성 높은 쪽을 우선 선택. 강제 기준(side_pref)와 비교."""
    def vis_sum(ids):
        return float(np.sum([lmk[i].visibility for i in ids]))
    L = [11,13,15,23,25,27]  # 어깨/팔/힙/무릎/발목(좌)
    R = [12,14,16,24,26,28]  # 우
    left_vis, right_vis = vis_sum(L), vis_sum(R)
    if side_pref == "LEFT":
        return "LEFT" if left_vis >= right_vis*0.8 else ("RIGHT" if right_vis>0 else "LEFT")
    else:
        return "RIGHT" if right_vis >= left_vis*0.8 else ("LEFT" if left_vis>0 else "RIGHT")

def get_xy(lmk, i, w, h):
    return np.array([lmk[i].x*w, lmk[i].y*h], dtype=np.float32)

def pelvis_tilt_deg(lmk, w, h):
    """양쪽 ASIS 근사: 좌/우 힙(23,24) 연결선의 수평 대비 기울기 각."""
    Lhip = get_xy(lmk, 23, w, h); Rhip = get_xy(lmk, 24, w, h)
    v = Rhip - Lhip
    horiz = np.array([1.0, 0.0])
    cosang = np.dot(v, horiz) / (np.linalg.norm(v)*np.linalg.norm(horiz)+1e-8)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))

def knee_valgus_ratio_front(lmk, w, h, side="LEFT"):
    """정면 촬영 시 무릎 안짖음 지표: 무릎 x가 발목-엉덩이 선의 안쪽으로 얼마나 들어왔는지 비율."""
    if side=="LEFT":
        hip, knee, ankle = 23, 25, 27
    else:
        hip, knee, ankle = 24, 26, 28
    H = get_xy(lmk, hip, w, h); K = get_xy(lmk, knee, w, h); A = get_xy(lmk, ankle, w, h)
    # 엉덩이-발목 선분의 x 범위 대비 무릎 x의 내측 이동량을 비율화
    x_min, x_max = min(H[0], A[0]), max(H[0], A[0])
    if x_max - x_min < 1:
        return 0.0
    if H[0] <= A[0]:  # 좌우 정렬 보정
        inward = max(0.0, x_min - K[0])
    else:
        inward = max(0.0, K[0] - x_max)
    return inward / (x_max - x_min)

# -----------------------------
# 3) 업로드
# -----------------------------
uf = st.file_uploader("스쿼트 영상 업로드(mp4, mov 등)", type=["mp4","mov","m4v","avi"])
col_prev, col_sum = st.columns([1,1])

# -----------------------------
# 4) 처리 루틴
# -----------------------------
if uf is not None:
    # 임시 저장
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uf.name).suffix)
    tmp.write(uf.read()); tmp.flush(); tmp.close()
    cap = cv2.VideoCapture(tmp.name)
    if not cap.isOpened():
        st.error("영상 열기 실패")
        st.stop()

    # MediaPipe Pose 초기화
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # 수집 버퍼
    frames, ts, knee_deg, hip_deg, torso_lean, hip_y, knee_y, ankle_y, valgus_ratio = [], [], [], [], [], [], [], [], []
    H, W, fps = None, None, cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 프레임 루프
    fidx = 0
    preview_holder = col_prev.empty()
    start_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if H is None:
            H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(rgb)
        if res.pose_landmarks is None:
            # 결측 프레임 처리: NaN 채움
            ts.append(fidx/fps)
            knee_deg.append(np.nan); hip_deg.append(np.nan); torso_lean.append(np.nan)
            hip_y.append(np.nan); knee_y.append(np.nan); ankle_y.append(np.nan)
            valgus_ratio.append(np.nan)
            frames.append(frame if show_annot else None)
            fidx += 1
            continue

        lmk = res.pose_landmarks.landmark
        side = select_side(lmk, side_pref)

        if side=="LEFT":
            hip_i, knee_i, ank_i, sh_i = 23,25,27,11
        else:
            hip_i, knee_i, ank_i, sh_i = 24,26,28,12

        Hxy = get_xy(lmk, hip_i, W, H)
        Kxy = get_xy(lmk, knee_i, W, H)
        Axy = get_xy(lmk, ank_i, W, H)
        Shxy= get_xy(lmk, sh_i, W, H)

        # 각도 계산
        knee_angle = angle(Hxy, Kxy, Axy)            # 무릎 굴곡 각(큰 값=신전)
        hip_angle  = angle(Shxy, Hxy, Kxy)           # 엉덩이 굴곡 근사
        torso_deg  = vec_angle_with_vertical(Shxy, Hxy)  # 상체 전경사

        # 정면 시 무릎 안짖음 비율
        vr = knee_valgus_ratio_front(lmk, W, H, "LEFT" if side=="LEFT" else "RIGHT") if view=="정면" else 0.0

        # 기록
        ts.append(fidx/fps)
        knee_deg.append(knee_angle)
        hip_deg.append(hip_angle)
        torso_lean.append(torso_deg)
        hip_y.append(Hxy[1]); knee_y.append(Kxy[1]); ankle_y.append(Axy[1])
        valgus_ratio.append(vr)

        # 주석 표시
        if show_annot:
            draw = frame.copy()
            for p in [Hxy, Kxy, Axy, Shxy]:
                cv2.circle(draw, tuple(p.astype(int)), 4, (0,255,0), -1)
            cv2.putText(draw, f"side:{side} knee:{knee_angle:0.1f} hip:{hip_angle:0.1f} torso:{torso_deg:0.1f}",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            preview_holder.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            frames.append(None)  # 미리보기만
        else:
            frames.append(None)

        fidx += 1

    cap.release(); pose.close()

    # -----------------------------
    # 5) 시계열 후처리 및 반복 검출
    # -----------------------------
    ts = np.array(ts)
    knee_deg = moving_avg(np.array(knee_deg, dtype=float), smooth_win)
    hip_deg  = moving_avg(np.array(hip_deg, dtype=float),  smooth_win)
    torso_lean = moving_avg(np.array(torso_lean, dtype=float), smooth_win)
    hip_y = moving_avg(np.array(hip_y, dtype=float), smooth_win)
    knee_y = moving_avg(np.array(knee_y, dtype=float), smooth_win)
    ankle_y = moving_avg(np.array(ankle_y, dtype=float), smooth_win)
    valgus_ratio = moving_avg(np.array(valgus_ratio, dtype=float), smooth_win)

    # 깊이 판정 신호
    if depth_rule == "힙이 무릎 아래":
        depth_ok = (hip_y - knee_y) > hip_below_knee_margin  # y는 아래로 클수록 아래
    else:
        depth_ok = (knee_deg <= (180 - knee_flex_bottom))    # 굴곡이 충분히 큼

    # 간단 FSM: Eccentric(하강) → Bottom → Concentric(상승) → Top
    state = "TOP"
    reps = []
    rep_start_idx = None
    last_bottom_idx = None

    for i in range(len(ts)):
        kd = knee_deg[i]
        if np.isnan(kd):
            continue

        if state == "TOP":
            if kd <= (180 - knee_flex_start):
                state = "DOWN"
                rep_start_idx = i
        elif state == "DOWN":
            if depth_ok[i]:
                state = "BOTTOM"
                last_bottom_idx = i
        elif state == "BOTTOM":
            if kd >= knee_extend_end:
                # 반복 종료
                rep_end_idx = i
                dur = ts[rep_end_idx] - ts[rep_start_idx] if rep_start_idx is not None else np.nan
                if pd.notna(dur) and dur >= min_rep_duration:
                    reps.append((rep_start_idx, last_bottom_idx or rep_end_idx, rep_end_idx))
                state = "TOP"
                rep_start_idx = None
                last_bottom_idx = None

    # -----------------------------
    # 6) 리포트 테이블 생성
    # -----------------------------
    rows = []
    for r, (i0, ib, i1) in enumerate(reps, start=1):
        # 구간 통계
        depth_metric = float(np.nanmax((hip_y[i0:i1+1] - knee_y[i0:i1+1])) if i1>i0 else np.nan)
        min_knee_deg = float(np.nanmin(knee_deg[i0:i1+1]))
        max_hip_flex = float(np.nanmax(180-hip_deg[i0:i1+1]))  # 굴곡 크기
        mean_torso   = float(np.nanmean(torso_lean[i0:i1+1]))
        wink_deg     = float(np.nanmax(np.diff(hip_y[i0:i1+1]))*-1 if i1>i0 else 0.0)  # 하강말기-상승초 골반 후방경사 근사
        mean_valgus  = float(np.nanmean(valgus_ratio[i0:i1+1])) if view=="정면" else np.nan
        duration     = float(ts[i1]-ts[i0])

        warn_depth = depth_metric <= hip_below_knee_margin if depth_rule=="힙이 무릎 아래" else ((180-min_knee_deg) < knee_flex_bottom)
        warn_torso = mean_torso >= torso_lean_warn
        warn_wink  = wink_deg >= wink_thresh_deg
        warn_valgus= (view=="정면") and (mean_valgus >= valgus_ratio_warn)

        rows.append({
            "rep": r,
            "start_s": round(ts[i0],3),
            "bottom_s": round(ts[ib],3),
            "end_s": round(ts[i1],3),
            "duration_s": round(duration,3),
            "depth_metric": round(depth_metric,2),
            "min_knee_deg": round(min_knee_deg,1),
            "max_hip_flex_deg": round(max_hip_flex,1),
            "mean_torso_lean_deg": round(mean_torso,1),
            "knee_valgus_ratio": None if np.isnan(mean_valgus) else round(mean_valgus,3),
            "WARN_depth": bool(warn_depth),
            "WARN_torso": bool(warn_torso),
            "WARN_wink": bool(warn_wink),
            "WARN_valgus": bool(warn_valgus)
        })

    rep_df = pd.DataFrame(rows)

    # -----------------------------
    # 7) 요약 및 표시
    # -----------------------------
    with col_sum:
        st.subheader("요약")
        total_reps = int(rep_df["rep"].count()) if not rep_df.empty else 0
        avg_dur = float(rep_df["duration_s"].mean()) if total_reps>0 else 0.0
        st.metric("반복수", total_reps)
        st.metric("평균 템포(초/회)", round(avg_dur,2))

        if total_reps>0:
            st.caption("경고 규칙: 깊이 부족, 과도한 전경사, 힙윙크, 정면 무릎 안짖음")
            st.dataframe(rep_df, use_container_width=True, height=360)

            # CSV 다운로드
            csv = rep_df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("결과 CSV 다운로드", data=csv, file_name="squat_report.csv", mime="text/csv")
        else:
            st.info("반복이 검출되지 않았습니다. 임계값 슬라이더를 조정하거나 촬영 시야를 확인하세요.")

    # -----------------------------
    # 8) 주의사항
    # -----------------------------
    st.markdown("---")
    st.markdown("**주의**")
    st.markdown(
        "- 측면 촬영에서는 무릎 안짖음(Valgus) 추정이 제한적입니다. 정면 시야에서만 사용하세요.\n"
        "- 힙윙크는 골반/허리 라인의 2D 변화로 근사합니다. 조명·의복·가림의 영향을 받습니다.\n"
        "- 깊이 판정은 ‘힙이 무릎 아래’ 또는 ‘무릎각’ 중 택1. 촬영 거리와 해상도에 따라 픽셀 기준을 조정하세요."
    )
else:
    st.info("스쿼트 영상을 업로드하세요.")
