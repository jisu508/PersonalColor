"""
backend 패키지 — 퍼스널컬러 진단의 '두뇌' (웹과 무관한 순수 Python 로직)

    backend/
    ├── color_utils.py    공통 도구: 색 공간 변환(BGR→Lab), 영역 지정, 평균색   [공용]
    ├── vision/           영상처리 파트: 사진 → 숫자(색)                         [영상처리]
    │   ├── lighting.py       2단계: 흰 종이 기준 조명 보정
    │   └── face_color.py     3단계: MediaPipe 얼굴 검출 → 피부·눈·머리 Lab
    ├── diagnosis/        알고리즘 파트: 숫자(색) → 판정                         [알고리즘]
    │   ├── season.py         4단계: 시즌 기준표와의 거리 → 퍼센티지
    │   └── confidence.py     5단계: 신뢰도 0~100%
    └── pipeline.py       6단계: 위 모듈을 한 줄로 연결 (사진 → 결과 dict)

규칙: vision/ 과 diagnosis/ 는 서로를 import 하지 않는다.
      둘을 잇는 것은 오직 pipeline.py 뿐. → 파트별로 독립 작업·테스트 가능.
"""
