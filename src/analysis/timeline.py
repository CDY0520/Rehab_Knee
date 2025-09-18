# -*- coding: utf-8 -*-
"""
파일명: src/analysis/timeline.py
기능: 포즈 npz(lm_x, lm_y, t_ms, valid)와 GT csv(event, time_ms)를 입력받아
      heel_y, toe_y, knee angle 시계열과 GT 이벤트를 타임라인 그래프로 저장한다.

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

    # 컬럼명 트림
    df.columns = [c.strip() for c in df.columns]
    if "event" not in df.columns or "time_ms" not in df.columns:
        raise ValueError(f"GT csv에 event,time_ms 컬럼이 필요: {list(df.columns)}")

    out = df[["event", "time_ms"]].copy()
    # 값 트림 + 대문자화
    out["event"] = out["event"].astype(str).str.strip().str.upper()
    out["time_ms"] = pd.to_numeric(out["time_ms"], errors="coerce")

    # 디버그: 유니크 라벨 출력
    print("[debug] GT unique events:", sorted(out["event"].unique().tolist()))

    # HS/TO/MS만 유지
    out = out[out["event"].isin(["HS", "TO", "MS"])].dropna(subset=["time_ms"]).reset_index(drop=True)
    return out

# 5) 시계열 추출 함수(extract_series)
def extract_series(lm_x, lm_y, valid, side="L", normalize=True):
    s = side.upper()
    i_heel = IDX[f"{s}_HEEL"]
    i_toe  = IDX[f"{s}_TOE"]
    i_knee = IDX[f"{s}_KNEE"]
    i_hip  = IDX[f"{s}_HIP"]
    i_ank  = IDX[f"{s}_ANK"]

    heel_y = lm_y[:, i_heel].copy()
    toe_y  = lm_y[:, i_toe ].copy()

    hip  = np.stack([lm_x[:, i_hip],  lm_y[:, i_hip]], axis=1)
    knee = np.stack([lm_x[:, i_knee], lm_y[:, i_knee]], axis=1)
    ank  = np.stack([lm_x[:, i_ank],  lm_y[:, i_ank]], axis=1)
    knee_angle = angle_abc(hip, knee, ank)

    for arr in (heel_y, toe_y, knee_angle):
        arr[~valid] = np.nan

    if not normalize:
        return heel_y, toe_y, knee_angle

    def norm01(a):
        lo, hi = np.nanpercentile(a, 1), np.nanpercentile(a, 99)
        if hi - lo < 1e-6:
            return np.zeros_like(a)
        return (a - lo) / (hi - lo)

    return heel_y, toe_y, norm01(knee_angle)

# 6) 플롯 함수(plot_timeline)
def plot_timeline(t_ms, heel_y, toe_y, knee_ang_n, gt_df, side, save_path):
    plt.figure(figsize=(14, 4.2))
    ax = plt.gca()

    ax.plot(t_ms, heel_y, label="Heel_y", lw=1.2)
    ax.plot(t_ms, toe_y,  label="Toe_y",  lw=1.2)
    ax.plot(t_ms, knee_ang_n, label="Knee angle (norm.)", lw=1.2)

    ax.set_title(f"Timeline overlay ({'LEFT' if side=='L' else 'RIGHT'})")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("normalized scale (0-1)")
    ax.grid(True, alpha=0.2)

    # GT 마커 표시
    if gt_df is not None and len(gt_df) > 0:
        ymin = np.nanmin([np.nanmin(heel_y), np.nanmin(toe_y), np.nanmin(knee_ang_n)])
        y0 = ymin - 0.05
        first = True
        for t, lab in zip(gt_df["time_ms"].values, gt_df["event"].str.upper().values):
            if first:
                ax.scatter([t], [y0], s=36, c="tab:blue", marker="o", zorder=6, label="GT")
                first = False
            else:
                ax.scatter([t], [y0], s=36, c="tab:blue", marker="o", zorder=6)
            ax.text(t, y0 - 0.03, lab, fontsize=9, ha="center", color="tab:red", zorder=6)

        # y축 하한 자동 확장
        cur_lo, cur_hi = ax.get_ylim()
        if cur_lo > y0 - 0.06:
            ax.set_ylim(y0 - 0.06, cur_hi)

    ax.legend(loc="lower right", ncol=4, fontsize=9, frameon=True)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

# 7) 메인 실행(main)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--side", default="L", choices=["L", "R"])
    ap.add_argument("--outdir", default="results/plots")
    args = ap.parse_args()

    lm_x, lm_y, t_ms, valid = load_pose_npz(args.npz)
    heel_y, toe_y, knee_ang_n = extract_series(lm_x, lm_y, valid, side=args.side, normalize=True)
    gt_df = load_gt_csv(args.gt)

    base = os.path.splitext(os.path.basename(args.npz))[0]
    save_path = os.path.join(args.outdir, f"{base}_{args.side}_timeline.png")
    print(f"[info] GT rows: {0 if gt_df is None else len(gt_df)}")
    print(f"[saved] {plot_timeline(t_ms, heel_y, toe_y, knee_ang_n, gt_df, args.side, save_path)}")

if __name__ == "__main__":
    main()
