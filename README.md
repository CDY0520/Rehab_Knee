# 신경계 재활 환자 특이 운동 패턴 자동 분석 파이프라인 개발

---

# 프로젝트 개요

스마트폰으로 촬영한 보행(gait) 영상을 입력받아 Mediapipe Pose 기반으로 관절 좌표를 추출하고,
무릎 중심 재활에 필요한 보행 이벤트 및 지표를 자동 분석하는 시스템이다.

핵심 목표:
무릎 과신전(Genū Recurvatum), Stiff-knee 이상 보행 패턴 자동 검출
정량적 이벤트 추출: HS(뒤꿈치 닿음), TO(발끝 이탈), MS(중간 디딤)
원격 재활 및 환자 자기 운동관리 지원

---

# 주요 기능

Pose 기반 보행 이벤트 분석 (src/events.py)

HS/TO/MS: heel_y − toe_y 차이 기반 규칙
GR(Genū Recurvatum): MS ± window 내 무릎 내부각 ≥ 임계 + knee_x 부호전환 검출
Stiff-knee: TO 시점 무릎 굴곡 부족

---

# 시각화

타임라인 그래프: Heel/Toe Y좌표 + Knee angle + 이벤트 라벨

---

# Streamlit 대시보드

📂 보행 영상 업로드 → 포즈 추출 & 이벤트 분석
📊 이벤트 기반 동작 분석 결과 → 치료사 코멘트 입력
📝 최종 리포트 다운로드

---

# 디렉토리 구조

Rehab_Knee/
│
├── app/
│   ├── pages/
│      ├── 01_영상업로드.py      # 보행 영상 업로드, 영상 품질검사
│      ├── 02_동작분석.py        # 보행 이벤트 분석 + 리포트
│       
├── src/
│   ├── qmetrics.py               # 영상 품질 지표 계산
│   ├── events.py                 # 보행 이벤트/지표 검출
│   ├── analysis/
│   │   ├── eval_events.py           # 라벨링 vs pred 비교 분석
│   │   ├── label_events.py          # openCV, 보행 이벤트 수동 라벨링
│   │   ├── run_gait_eval.py         # 보행 이벤트 pred
│   │   ├── timeline.py              # npz 파일 활용해서 타임라인 그래프
│   │   ├── timeline_knee_x.py       # 기존 타임라인 그래프 + knee x좌표 추가 그래프
│   │   └── viz_eval_results.py      # pred 비교 분석 결과 시각화 및 평가
│   └── pose_probe.py             # Mediapipe 포즈 추출 래퍼
│
├── results/
│   ├── keypoints/                # npz 포즈 데이터
│   ├── plots/                    # 분석 그래프
│   └── reports/                  # 리포트(json/csv/txt)
│
├── requirements.txt
└── README.md

---

# 설치 및 실행

1) 환경 세팅
 - git clone https://github.com/CDY0520/Rehab_Knee.git
 - cd Rehab_Knee
 - python -m venv .venv
 - source .venv/bin/activate   # Windows: .venv\Scripts\activate
 - pip install -r requirements.txt
2) Mediapipe 포즈 추출
 - python src/pose_probe.py --video data/samples/sample_walk_normal.mp4 --out results/keypoints/sample_walk_normal.npz
3) 보행 이벤트 분석 (CLI)
 - python src/events.py --npz results/keypoints/sample_walk_normal.npz --save-json --save-csv
4) 대시보드 실행
 - streamlit run app/pages/01_영상업로드.py
 - streamlit run app/pages/02_.py

---

# 사용 방법
1) Kinovea에서 데이터 추출
 - Kinovea로 보행/STS 영상 불러오기
 - 마커링 후 CSV/좌표 데이터 Export
2) 보행/STS 분석 실행
 - gait 분석: python -m src.gait_events --video data/samples/sample_walk.mp4 --side left --out results/sample_gait_left.json
 - STS 분석: python -m src.sts_events --video data/samples/sample_sts.mp4 --side left --out results/sample_sts_left.json
3) RAG 인덱싱 + LLM 피드백
 - RAG 인덱싱 (결과 JSON을 검색 문서화): python -m rag.indexer --input results/
 - LLM 요약/피드백 생성: python -m rag.generate --input results/sample_sts_left.json

---

# 결과 예시
1) 출력 파일: results/ 폴더 내 JSON, 시각화 이미지 저장
2) 예시 지표
 - 보행: 보폭, 보행주기, stance/swing 비율
 - STS: 일어서기 소요시간, 무릎 굴곡/신전 각도 변화
 - 무릎/발목: Kinovea 추출 기반 ROM, 기울기 등
3)AI 피드백: RAG + LLM 기반 환자별 요약 및 피드백 리포트

---

# 라이선스
MIT License
