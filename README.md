# Rehab_Knee Project
환자 보행 및 Sit-to-Stand(STS) 동작을 분석하고, 무릎 중심의 기능적 지표를 추출하는 프로젝트입니다.  
Kinovea 등 무료 분석 도구를 통해 추출한 데이터를 기반으로, RAG + LLM을 활용한 요약/피드백 기능까지 제공합니다.

---

# 프로젝트 개요
목표: 신경계 손상 환자의 무릎 및 발목 움직임을 중심으로 보행/STS 기능 분석
분석 도구: [Kinovea](https://www.kinovea.org) + Mediapipe 기반 포즈 추출
AI 적용: RAG 기반 LLM으로 결과 요약 및 환자별 피드백 생성
확장성: 환자 운동 모니터링 및 재활 지원 서비스로 발전 가능

---

# 디렉토리 구조
```bash
Rehab_Knee/
│── data/               # 원천 데이터 (raw), 전처리 (processed), 샘플 (samples)
│── results/            # 분석 결과 (json, figures, keypoints)
│── src/                # 주요 Python 코드 (config, adapters 등)
│── rag/                # RAG 인덱싱 및 프롬프트 템플릿
│── notebooks/          # 실험용 Jupyter 노트북
│── requirements.txt    # 패키지 의존성
│── README.md           # 프로젝트 문서

---

# 설치 방법
1) 저장소 클론
 - git clone https://github.com/CDY0520/Rehab_Knee.git
 - cd Rehab_Knee
2) 가상환경 생성 (선택)
 - python -m venv .venv
 - source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
3) 의존성 설치
 - pip install -r requirements.txt

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