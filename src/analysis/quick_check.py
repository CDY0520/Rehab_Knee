# 목적: sample_walk 신호 품질과 이벤트 후보를 숫자로 즉시 점검
from pathlib import Path
import numpy as np, json

npz = Path("results/keypoints/sample_walk.npz")
D = np.load(npz, allow_pickle=True)
lm_x, lm_y, lm_v = D["lm_x"], D["lm_y"], D["lm_v"]
t_ms = D["t_ms"]; valid = D["valid"]
meta = json.loads(str(D["meta"].item() if hasattr(D["meta"], "item") else D["meta"]))
fps = meta.get("fps", 30)

# Mediapipe 인덱스
L_ANKLE, R_ANKLE = 27, 28
L_FOOT, R_FOOT = 31, 32

def stats(name, y):
    dy = np.diff(y)
    print(f"[{name}] y.min={y.min():.4f}, y.max={y.max():.4f}, amp={y.max()-y.min():.4f}, "
          f"std={y.std():.4f}, |dy|mean={np.abs(dy).mean():.5f}")

print(f"frames={len(t_ms)}, valid_ratio={valid.mean():.3f}, fps={fps}")
# 좌/우 발목과 발끝 모두 확인
stats("L_ankle", lm_y[:, L_ANKLE])
stats("R_ankle", lm_y[:, R_ANKLE])
stats("L_toe",   lm_y[:, L_FOOT])
stats("R_toe",   lm_y[:, R_FOOT])

# 이동평균 3, 속도 부호 전환 개수로 후보 카운트
def movavg(x, k=3):
    if k<=1: return x
    import numpy as np
    w = np.ones(k)/k
    return np.convolve(x, w, mode="same")

def zero_cross_count(y, k=3):
    import numpy as np
    v = np.gradient(movavg(y, k))  # dt 생략 상대값만
    s = np.sign(v)
    return int(np.sum((s[:-1] > 0) & (s[1:] < 0))), int(np.sum((s[:-1] < 0) & (s[1:] > 0)))

for name, y in [("L_toe", lm_y[:, L_FOOT]), ("R_toe", lm_y[:, R_FOOT])]:
    n_posneg, n_negpos = zero_cross_count(y, k=3)
    print(f"{name}: +->-={n_posneg}, -->+= {n_negpos}")
