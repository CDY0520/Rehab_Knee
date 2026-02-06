# 신경계 재활 환자 특이 운동 패턴 자동 분석  

---

# 주요 목표  
*이벤트 미검출이 코드 오류인지, 비정상 보행 패턴 때문인지 해석 가능성 확인*  
관절 포인트·이벤트 분석 기반 무릎 관절 지표 자동 산출 및 시계열 분석  
  
규칙 기반 이벤트 검출 알고리즘을 다양한 환자 보행 영상에 적용하여, 검출 실패의 원인을 임상적으로 해석하는 것이 주 목적이다.  
일반적인 알고리즘 성능 향상이나 코드 일반화가 아닌, 이벤트가 검출되지 않은 이유가 코드 오류인지, 환자의 비정상 보행 패턴 때문인지를 분석하는데 초점을 두었다.  
즉, 영상 기반 포즈 추정 기술을 직접 적용해 환자 보행을 분석하며 규칙 기반 이벤트 검출의 임상적 한계와 패턴 해석 감각을 익히는 것을 목표로 한다.  
  
---

# 핵심 기여  
실제 환자 영상 활용 MediaPipe 관절 포인트 추출  
영상 처리 및 시계열 패턴 분석  

---

# 프로젝트 접근 방법  
Pose 좌표 시계열(heel–toe y, knee angle) 기반 이벤트 규칙 정의  
이벤트 검출 실패 사례를 정상/비정상 보행 패턴 관점에서 재해석  
무릎 과신전(Genu Recurvatum), 무릎 강직(Stiff-knee) 등 비정상 보행 패턴 유형화 수행  

---

# 핵심 이벤트  
무릎 과신전(Genū Recurvatum), Stiff-knee 이상 보행 패턴 자동 검출  
정량적 이벤트 추출: HS(뒤꿈치 닿음), TO(발끝 이탈), MS(중간 디딤)  

---

# 주요 분석 내용 요약  
  
1) 환자 개별 영상 내 보행 패턴 분석  
동일 환자 내에서의 시계열 변화(heel_y, toe_y, knee_angle)를 분석하여 주기성, 진폭, 패턴을 해석  
정면/측면 구간을 구분하고, 측면 구간 중심으로 유효 데이터를 평가  
  
2)이벤트 미검출 원인 임상적 해석  
발목 수의적 움직임이 거의 없는 경우(heel_y–toe_y 진폭이 매우 작음)는 이벤트 검출 실패가 정상적인 결과임을 확인  
무릎 과신전(genu recurvatum) 등 특정 패턴이 없는 경우에도 미검출은 코드 오류가 아님을 명시  
  
3) 환자군별 특징적 보행 패턴 유형화  
Ankle-inactive, Stiff-knee, Asymmetric 등으로 분류하여 규칙 기반 이벤트 검출의 한계와 임상적 활용 가능성을 함께 평가  
  
5) 임상 기반 알고리즘 해석 프레임 구축  
기술적 결과를 임상적 의미로 연결하는 분석 절차를 제시  
향후 모델 성능지표가 아닌 ‘임상적 타당성’을 평가할 수 있는 프레임워크로 확장 가능  
  
---

# 주요 기능  
Pose 기반 보행 이벤트 분석 (src/events.py)  
HS/TO/MS: heel_y − toe_y 차이 기반 규칙  
GR(Genū Recurvatum): MS ± window 내 무릎 내부각 ≥ 임계 + knee_x 부호전환 검출  
Stiff-knee: TO 시점 무릎 굴곡 부족  

---

# Streamlit 대시보드  
📂 보행 영상 업로드 → 포즈 추출 & 이벤트 분석  
📊 이벤트 기반 동작 분석 결과 → 치료사 코멘트 입력  
📝 최종 리포트 다운로드  

---

# 주요 성과  
이벤트 미검출 사례 다수가 비정상 보행 패턴에 기인함을 확인  
규칙 기반 검출의 한계를 임상적 맥락 설명으로 보완 → 임상 해석 중심  

---

# 인사이트  
영상 기반 보행 분석의 전제는 영상 품질 관리 → 이후 분석의 신뢰성과 연관됨  
보행 분석의 가치는 정상 검출이 아니라 비정상 데이터가 사용자에게 어떤 의미를 가지는지 해석하는 것에 있음  

---

# 디렉토리 구조  

```
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
```

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
 - streamlit run app/pages/02_동작분석.py  
  
---

# 결과 예시  

```
1) CLI 요약
   [LEFT] HS n=4, TO n=4, MS n=4, GR n=3, SK n=4
       knee_max_inner=179.6°, knee_min_inner=141.0°
   [RIGHT] HS n=3, TO n=3, MS n=3, GR n=3, SK n=3
       knee_max_inner=179.6°, knee_min_inner=156.8°

2) streamlit 리포트
   “왼쪽 뒤꿈치 닿음(HS): 4회 발생했습니다.”
   “⚠️ 왼쪽 무릎: 과신전이 관찰됩니다.”
   “⚠️ 왼쪽 무릎: 다리를 앞으로 내딛을 때 무릎 굽힘이 부족합니다.”
```

---
