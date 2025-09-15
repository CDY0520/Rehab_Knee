"""
파일명: src/analysis/label_events.py

설명:
  - 보행 영상(mp4)을 프레임 단위로 탐색하며 HS/TO 이벤트를 수동 라벨링한다.
  - L/R 사이드 전환, 단축키 입력으로 이벤트 기록 후 CSV 저장한다.
  - 출력:
      · results/gt/{video_id}_L.csv
      · results/gt/{video_id}_R.csv
      (컬럼: video_id,side,event,time_ms,frame)

사용법 예시:
  python src/analysis/label_events.py

블록 구성:
  0) import 및 경로 설정
  1) 영상 로드 및 상태 초기화
  2) 라벨링 인터페이스 (OpenCV 창 + 단축키)
  3) 라벨 저장(CSV 출력)
  4) 메인 루프 (video_id별 실행)
"""

import cv2, csv
from pathlib import Path

VIDEO_MAP = {
    "normal":   "data/samples/sample_walk_normal.mp4",
    "hyperext": "data/samples/sample_walk.mp4"
}
GT_DIR = Path("results/gt"); GT_DIR.mkdir(parents=True, exist_ok=True)

def label(video_id: str):
    vpath = VIDEO_MAP[video_id]
    cap = cv2.VideoCapture(vpath)
    if not cap.isOpened():
        print("open fail:", vpath); return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    side = "L"
    cur = 0
    marks = []  # {"side","event","frame","time_ms"}

    def redraw():
        cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
        ok, frame = cap.read()
        if not ok: return
        disp = frame.copy()
        txt = f"{video_id} | side:{side} | frame:{cur}/{total-1} | fps:{fps:.1f}"
        cv2.putText(disp, txt, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        y=60
        for m in marks[-6:]:
            cv2.putText(disp, f"{m['side']} {m['event']} @f{m['frame']} t{m['time_ms']}ms",
                        (20,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            y+=22
        cv2.imshow("label", disp)

    while True:
        redraw()
        k = cv2.waitKey(0) & 0xFF

        if k in [ord('q'), 27]: break
        elif k==ord('1'): side="L"
        elif k==ord('2'): side="R"
        elif k==ord('j'): cur=max(0, cur-1)
        elif k==ord('k'): cur=min(total-1, cur+1)
        elif k==ord('J'): cur=max(0, cur-5)
        elif k==ord('K'): cur=min(total-1, cur+5)
        elif k==ord('u') and marks: marks.pop()
        elif k==ord('h') or k==ord('t'):
            ev = "HS" if k==ord('h') else "TO"
            t_ms = int(round(cur / fps * 1000.0))
            marks.append({"side": side, "event": ev, "frame": cur, "time_ms": t_ms})
        elif k==ord('s'):
            # 사이드별 저장
            for s in ["L","R"]:
                out = GT_DIR / f"{video_id}_{s}.csv"
                rows = [m for m in marks if m["side"]==s]
                if not rows: continue
                with out.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=["video_id","side","event","time_ms","frame"])
                    w.writeheader()
                    for r in rows:
                        w.writerow({"video_id": video_id, **r})
                print("[saved]", out)

    cap.release(); cv2.destroyAllWindows()

if __name__=="__main__":
    for vid in ["normal","hyperext"]:
        try: label(vid)
        except Exception as e: print(e)
