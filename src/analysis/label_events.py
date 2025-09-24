"""
파일명: src/analysis/label_events.py

설명:
  - 보행(Gait)과 STS 이벤트를 한 스크립트에서 수동 라벨링한다.
  - Gait: HS, TO, MS, GENU_RECURVATUM
  - STS : seat_off(SO), full_stand(FS)  ※ STS는 사이드 'C'로 저장
  - 저장 형식(csv): video_id,side,event,time_ms,frame  → results/gt/{video_id}_{side}.csv

사용법:
  python src/analysis/label_events.py

단축키:
  [공통]  j/J/k/K = 프레임 -1/-5/+1/+5,  u = 되돌리기,  s = 저장,  q/Esc = 종료,  z = 모드(Gait/STS) 토글
  [Gait]  1=L, 2=R,  h=HS,  t=TO,  m=MS,  g=GENU_RECURVATUM
  [STS ]  o=Seat-off,  f=Full-stand   (사이드는 자동으로 'C')

블록 구성:
  0) import·경로 설정 및 영상 매핑
  1) 라벨 구조·유틸(시간 환산, 저장)
  2) OpenCV 인터페이스(모드 토글, 사이드, 단축키 처리)
  3) 메인: normal → hyperext 순회
"""

import cv2, csv
from pathlib import Path

# 0) 경로·영상 매핑 ------------------------------------------------------------
VIDEO_MAP = {
    "normal":   "data/samples/sample_walk_normal.mp4",
    "hyperext": "data/samples/sample_walk_hyper.mp4", # 루프본(과신전)
}
OUT_DIR = Path("results/gt"); OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1) 라벨 구조·유틸 ------------------------------------------------------------
def _save_marks(video_id: str, marks: list[dict]):
    # Gait: L/R, STS: C를 각각 파일로 저장
    by_side = {}
    for m in marks:
        by_side.setdefault(m["side"], []).append(m)
    for side, rows in by_side.items():
        out = OUT_DIR / f"{video_id}_{side}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_id","side","event","time_ms","frame"])
            w.writeheader()
            for r in rows:
                w.writerow({"video_id": video_id, **r})
        print("[saved]", out)

def _ms_from_frame(frame: int, fps: float) -> int:
    return int(round(frame / max(fps, 1e-6) * 1000.0))

# 2) OpenCV 라벨러 --------------------------------------------------------------
def label_video(video_id: str):
    vpath = VIDEO_MAP[video_id]
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print("open fail:", vpath); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mode = "gait"   # 'gait' or 'sts'
    side = "L"      # gait 모드에서만 사용. sts는 항상 'C'
    cur  = 0
    marks: list[dict] = []  # {"side","event","frame","time_ms"}

    def _put_text(img, text, y):
        cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

    def redraw():
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = cap.read()
        if not ok: return
        disp = frame.copy()

        mode_text = f"MODE: {mode.upper()}   FPS:{fps:.1f}   frame:{cur}/{total-1}"
        side_text = f"SIDE: {side if mode=='gait' else 'C'}"
        help_text = ("[Gait] 1=L 2=R | h=HS t=TO m=MS g=GR  | j/J/k/K=-1/-5/+1/+5"
                     "   [STS] o=SO f=FS   | u=undo s=save z=mode q=quit")

        _put_text(disp, f"{video_id} | {mode_text} | {side_text}", 28)
        _put_text(disp, help_text, 56)

        y = 84
        for m in marks[-8:]:
            _put_text(disp, f"{m['side']} {m['event']} @f{m['frame']}  {m['time_ms']}ms", y)
            y += 22

        cv2.imshow("label", disp)

    while True:
        redraw()
        k = cv2.waitKey(0) & 0xFF

        if k in [ord('q'), 27]:
            break

        # 모드 토글
        if k == ord('z'):
            mode = "sts" if mode == "gait" else "gait"
            continue

        # 프레임 이동
        if   k == ord('j'): cur = max(0, cur-1)
        elif k == ord('k'): cur = min(total-1, cur+1)
        elif k == ord('J'): cur = max(0, cur-5)
        elif k == ord('K'): cur = min(total-1, cur+5)

        # 되돌리기
        elif k == ord('u') and marks:
            marks.pop()

        # 사이드 선택(가이트 전용)
        elif mode == "gait" and k == ord('1'):
            side = "L"
        elif mode == "gait" and k == ord('2'):
            side = "R"

        # 이벤트 기록
        elif mode == "gait" and k in [ord('h'), ord('t'), ord('m'), ord('g')]:
            ev = {ord('h'):"HS", ord('t'):"TO", ord('m'):"MS", ord('g'):"HY.EXT."}[k]
            t_ms = _ms_from_frame(cur, fps)
            marks.append({"side": side, "event": ev, "frame": cur, "time_ms": t_ms})

        elif mode == "sts" and k in [ord('o'), ord('f')]:
            ev = {ord('o'):"seat_off", ord('f'):"full_stand"}[k]
            t_ms = _ms_from_frame(cur, fps)
            marks.append({"side": "C", "event": ev, "frame": cur, "time_ms": t_ms})

        # 저장
        elif k == ord('s'):
            _save_marks(video_id, marks)

    cap.release(); cv2.destroyAllWindows()

# 3) 메인 -----------------------------------------------------------------------
if __name__ == "__main__":
    for vid in ["normal", "hyperext"]:
        try:
            label_video(vid)
        except Exception as e:
            print(e)
