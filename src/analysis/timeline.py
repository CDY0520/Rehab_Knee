# -*- coding: utf-8 -*-
"""
파일명: src/analysis/timeline.py

기능: 포즈 npz(lm_x, lm_y, t_ms, valid)와 GT csv(event, time_ms)를 입력받아
      heel_y, toe_y, knee angle 시계열과 GT 이벤트를 타임라인 그래프로 저장한다.

코드 실행
  python src/analysis/timeline.py --pose results/keypoints/sample_walk3.npz --gt results/gt/hyperext_R.csv --side R --outdir results/plots

블록 구성
 0) 라이브러리 임포트
 1) Mediapipe 인덱스 정의
 2) 각도 계산 유틸(angle_abc)
 3) npz 로드 함수(load_pose_npz)
 4) GT 로드 함수(load_gt_csv)
 5) 시계열 추출 함수(extract_series)
 6) 플롯 함수(plot_timeline)
 7) 메인 실행(main)
"""

# 0) 라이브러리 임포트
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1) Mediapipe 인덱스 정의
IDX = dict(
    L_HIP=23, R_HIP=24,
    L_KNEE=25, R_KNEE=26,
    L_ANK=27, R_ANK=28,
    L_HEEL=29, R_HEEL=30,
    L_TOE=31, R_TOE=32,
)

# 2) 각도 계산 유틸(angle_abc)
def angle_abc(a, b, c):
    """벡터 BA와 BC 사이 각도(도)"""
    ba = a - b
    bc = c - b
    nba = np.linalg.norm(ba, axis=-1) + 1e-8
    nbc = np.linalg.norm(bc, axis=-1) + 1e-8
    cosv = np.clip(np.sum(ba * bc, axis=-1) / (nba * nbc), -1, 1)
    return np.degrees(np.arccos(cosv))

# 3) npz 로드 함수(load_pose_npz)
def load_pose_npz(path_npz):
    d = np.load(path_npz, allow_pickle=True)
    if not {"lm_x", "lm_y"}.issubset(d.files):
        raise ValueError(f"npz 키 부족: {d.files}")
    lm_x = d["lm_x"].astype(float)
    lm_y = d["lm_y"].astype(float)
    if "t_ms" in d:
        t_ms = d["t_ms"].astype(float)
    else:
        fps = 30.0
        if "meta" in d and isinstance(d["meta"].item(), dict):
            fps = float(d["meta"].item().get("fps", 30.0))
        t_ms = np.arange(lm_x.shape[0]) / fps * 1000.0
    valid = d["valid"].astype(bool) if "valid" in d else np.ones(lm_x.shape[0], bool)
    return lm_x, lm_y, t_ms, valid

# 4) GT 로드 함수(load_gt_csv)
def load_gt_csv(path_csv):
    if not path_csv or not os.path.isfile(path_csv):
        return None
    df = pd.read_csv(path_csv)

    # 컬럼명 정리
    df.columns = [c.strip() for c in df.columns]
    if "event" not in df.columns or "time_ms" not in df.columns:
        raise ValueError(f"GT csv에 event,time_ms 컬럼이 필요: {list(df.columns)}")

    out = df[["event", "time_ms"]].copy()
    # 값 정리
    out["event"] = out["event"].astype(str).str.strip().str.upper()
    out["time_ms"] = pd.to_numeric(out["time_ms"], errors="coerce")

    # 디버그: 유니크 라벨
    print("[debug] GT unique events:", sorted(out["event"].unique().tolist()))

    # HS/TO/MS/과신전(HY.EXT.) 유지
    out = out[out["event"].isin(["HS", "TO", "MS", "HY.EXT."])].dropna(subset=["time_ms"]).reset_index(drop=True)
    return out

# 5) 시계열 추출 함수(extract_series)
def extract_series(lm_x, lm_y, valid, side="L", normalize_knee=False):
    s = side.upper()
    i_heel = IDX[f"{s}_HEEL"]; i_toe  = IDX[f"{s}_TOE"]
    i_knee = IDX[f"{s}_KNEE"]; i_hip  = IDX[f"{s}_HIP"]; i_ank = IDX[f"{s}_ANK"]

    heel_y = lm_y[:, i_heel].astype(float).copy()
    toe_y  = lm_y[:, i_toe ].astype(float).copy()

    hip  = np.stack([lm_x[:, i_hip],  lm_y[:, i_hip]],  axis=1).astype(float)
    knee = np.stack([lm_x[:, i_knee], lm_y[:, i_knee]], axis=1).astype(float)
    ank  = np.stack([lm_x[:, i_ank],  lm_y[:, i_ank]],  axis=1).astype(float)
    knee_angle_deg = angle_abc(hip, knee, ank)  # 도 단위

    # invalid 프레임 NaN
    for arr in (heel_y, toe_y, knee_angle_deg):
        arr[~valid] = np.nan

    if not normalize_knee:
        return heel_y, toe_y, knee_angle_deg

    # 필요 시 각도만 0-1 정규화 옵션 유지
    lo, hi = np.nanpercentile(knee_angle_deg, 1), np.nanpercentile(knee_angle_deg, 99)
    knee_n = np.zeros_like(knee_angle_deg) if hi - lo < 1e-6 else (knee_angle_deg - lo) / (hi - lo)
    return heel_y, toe_y, knee_n

# 6) 플롯 함수(plot_timeline) - 오른쪽 y축에 무릎 각도(도) 표시
def plot_timeline(t_ms, heel_y, toe_y, knee_ang_deg, gt_df, side, save_path):
    """타임라인 플롯을 저장하고 경로를 반환"""
    plt.figure(figsize=(14, 4.6))
    ax = plt.gca()

    # 왼쪽 축: y좌표 시계열
    ln1, = ax.plot(t_ms, heel_y, label="Heel_y", lw=1.2)
    ln2, = ax.plot(t_ms, toe_y,  label="Toe_y",  lw=1.2)

    ax.set_title(f"Timeline overlay ({'LEFT' if side=='L' else 'RIGHT'})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("y-coordinate")
    ax.grid(True, alpha=0.25)

    # 왼쪽 축 범위 고정: 아래=1.0, 위=0.0
    ax.set_ylim(1.0, 0.0)

    # 오른쪽 축: 무릎 각도(도)
    ax2 = ax.twinx()
    ln3, = ax2.plot(t_ms, knee_ang_deg, label="Knee angle (deg)",
                    lw=1.6, linestyle="-", color="green")
    ax2.set_ylabel("Knee angle (deg)")

    # 각도 축 자동 범위(과도한 아웃라이어 방지용 백분위 기반)
    k_lo = np.nanpercentile(knee_ang_deg, 1)
    k_hi = np.nanpercentile(knee_ang_deg, 99)
    if np.isfinite(k_lo) and np.isfinite(k_hi) and k_hi - k_lo > 1e-3:
        ax2.set_ylim(k_lo - 3, k_hi + 3)

    # GT 마커는 왼쪽 축 기준으로 표시
    sc_gt = None
    if gt_df is not None and len(gt_df) > 0:
        cur_lo, cur_hi = ax.get_ylim()
        y0 = cur_lo + (cur_hi - cur_lo) * 0.05
        first = True
        for t, lab in zip(gt_df["time_ms"].values, gt_df["event"].str.upper().values):
            if first:
                sc_gt = ax.scatter([t], [y0], s=40, c="tab:blue", marker="o",
                                   zorder=6, label="GT")
                first = False
            else:
                ax.scatter([t], [y0], s=40, c="tab:blue", marker="o", zorder=6)
            ax.text(t, y0 + (cur_hi - cur_lo) * 0.03, lab,
                    fontsize=9, ha="center", va="bottom",
                    color="tab:red", zorder=6)

    # 범례 결합
    handles = [ln1, ln2, ln3] + ([sc_gt] if sc_gt is not None else [])
    labels = [h.get_label() for h in handles]
    ax.legend(handles, labels, loc="upper right", ncol=len(handles), fontsize=9, frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    return save_path

# 7) 메인 실행(main)
def main():
    # argparse: --pose와 --npz 둘 다 지원. 저장만 수행.
    ap = argparse.ArgumentParser(description="Pose/GT 타임라인 플롯 자동 저장")
    ap.add_argument("--pose", "--npz", dest="npz", required=True, help="입력 pose npz 경로")
    ap.add_argument("--gt", default=None, help="입력 GT csv 경로")
    ap.add_argument("--side", default="L", choices=["L", "R"], help="분석 측면")
    ap.add_argument("--outdir", default="results/plots", help="저장 디렉토리")
    args = ap.parse_args()

    # 로드
    lm_x, lm_y, t_ms, valid = load_pose_npz(args.npz)
    print(f"[info] pose npz 로드 완료: {args.npz}, shape={lm_x.shape}")
    gt_df = load_gt_csv(args.gt) if args.gt else None
    print(f"[info] GT rows: {0 if gt_df is None else len(gt_df)}")

    # 시계열
    heel_y, toe_y, knee_ang = extract_series(lm_x, lm_y, valid, side=args.side, normalize_knee=False)

    # 저장 경로
    base = os.path.splitext(os.path.basename(args.npz))[0]
    os.makedirs(args.outdir, exist_ok=True)
    save_path = os.path.join(args.outdir, f"{base}_{args.side}_timeline.png")

    # 플롯 저장 (보조 y축 포함)
    saved = plot_timeline(t_ms, heel_y, toe_y, knee_ang, gt_df, args.side, save_path)
    print(f"[saved] {saved}")

if __name__ == "__main__":
    main()
