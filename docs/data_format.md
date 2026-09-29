# 데이터 형식 (초안 — 0주차 회의에서 확정)

파트끼리 주고받는 데이터 모양을 여기서 정한다. **바꿀 때는 이 문서부터 고치고 단톡에 공유.**

---

## ① 영상처리 → 알고리즘 : `extract_face_colors(img).to_dict()`

작성: 영상처리 (이지수·여) · 구현 완료 (`backend/vision/face_color.py`)

```python
{
  "skin": [74.4, 15.2, 13.3],      # 피부 대표 Lab (두 볼 + 이마[볼과 비슷할 때만])  / 실패 시 None
  "eye":  [19.5, 6.7, 12.8],       # 눈동자(홍채) 대표 Lab                          / 실패 시 None
  "hair": [25.2, 3.5, 5.3],        # 머리카락 Lab (이마 위 + 앞머리 영역 중 피부보다 어두운 픽셀) / 실패 시 None

  "regions": {                     # 부위별 상세 (디버그·신뢰도용)
    "cheek_right": {"lab": [74.0, 14.8, 13.4], "n_total": 8587, "n_used": 6910, "used": True,  "note": ""},
    "cheek_left":  {...},
    "forehead":    {"lab": [21.9, 3.7, 6.0], ..., "used": False, "note": "볼과 색 차이 ΔE 54.2 → 앞머리·그림자로 보고 제외"},
    "iris_a": {...}, "iris_b": {...},
    "hair": {...}
  },
  "quality": {                     # 사진 품질 (5단계 신뢰도 재료)
    "brightness": 74.4,            # 피부 밝기 L (0~100)
    "sharpness": 34.0,             # 선명도 (얼굴을 가로 400px 로 맞춘 뒤 라플라시안 분산, 10 미만이면 흐림)
    "face_width_ratio": 0.243,     # 얼굴 폭 / 사진 폭
    "asymmetry": 0.061             # 좌우 비대칭 (0.15 넘으면 옆얼굴)
  },
  "face": {"box": [x1, y1, x2, y2], "width_ratio": 0.24, "num_faces": 1},
  "warnings": []                   # 사람이 읽는 경고 문구 목록
}
```

- Lab 단위: **L 0~100 (밝기), a 초록(-)~빨강(+), b 파랑(-)~노랑(+)**
- 알고리즘 파트는 `skin`, `eye`, `hair` 세 값만 쓰면 된다. `None` 이 올 수 있으니 반드시 처리.
- `regions`, `face`, `warnings` 는 5단계 신뢰도 계산 재료.

## ② 조명 보정 → 신뢰도 : `white_balance(img, box).to_dict()`

작성: 영상처리 · 구현 완료 (`backend/vision/lighting.py`)

```python
{
  "gains_bgr": [1.457, 1.118, 1.03],   # 채널별 보정 배율
  "paper_lab": [90.9, -1.2, 28.3],     # 보정 전 종이 색 (a·b 가 0에서 멀수록 조명 색이 강했음)
  "cast_strength": 28.3,               # 조명 색끼 세기
  "paper_clip_ratio": 0.0, "paper_std": 4.0, "newly_clipped_ratio": 0.098,
  "warnings": []
}
```

## ③ 파이프라인 → 웹 : `run_pipeline()` 결과  (구현 완료)

`backend/pipeline.py` 의 `run_pipeline(img, guide_box=...)` 가 돌려주는 형식입니다.
자세한 예시와 연결 코드는 **docs/web_integration.md** 참고.

```python
{
  "ok": True,
  "percentages": [{"key": "winter_cool", "name": "겨울 쿨", "value": 50.0, "color": "#2B5299"}, ...],
  "best_group": "겨울 쿨",          # 1위 시즌 한글 이름
  "reliability": 80,               # 신뢰도 0~100
  "reliability_msg": "참고용으로 충분",
  "message": "두 시즌 경계에 있어 ...",   # 경계 케이스 해석
  "warnings": [...],               # 사용자 안내문
  # 아래는 디버그·분석용
  "seasons": {"winter_cool": 50.0, ...}, "top": "winter_cool", "warm_cool": "cool",
  "confidence": {...}, "colors": {...}, "quality": {...}, "lighting": {...},
  "season_detail": {...}, "method": "b"
}
```

## ④ 시즌 판정 기준값 : `data/reference/season_reference.json`

코드를 고치지 않고 이 파일의 숫자만 바꿔 실험합니다.

- `warm_cool.method` : `b`(기본) / `paper_d`(논문①) / `smtc`(논문③)
- `warm_cool.b.threshold` : 피부 b값 경계. **지금은 임시값 16.0**
- `second_axis` : 4계절로 나눌 때 쓰는 밝기·채도 기준 (아직 검증 전)
- 라벨 붙은 사진이 모이면 `python tools/fit_threshold.py <폴더>` 로 경계값을 다시 계산해 이 파일에 적습니다.

## (참고) 옛 초안 — 시즌 판정 결과

```python
{"spring_warm": 12.0, "summer_cool": 5.0, "autumn_warm": 8.0, "winter_cool": 75.0}   # 합계 100
```

## ④ 파이프라인 → 웹 : `run_pipeline(...)` 결과 (6단계, 초안)

```python
{
  "seasons": {"spring_warm": 12.0, "summer_cool": 5.0, "autumn_warm": 8.0, "winter_cool": 75.0},
  "top": "winter_cool",
  "confidence": 82,                 # 0~100
  "message": "전형적인 겨울 쿨톤",   # 경계 케이스 해석 문구
  "colors": {"skin": [...], "eye": [...], "hair": null},
  "warnings": ["..."]               # 재촬영 안내 등 (조명 + 얼굴 경고 합친 것)
}
```

## 회의에서 정할 것
- [ ] ③ 시즌 이름 키 (`spring_warm` 등) 확정
- [ ] 퍼센티지를 0~100 으로 줄지, 0~1 로 줄지
- [ ] `hair` 가 None 일 때 알고리즘이 피부·눈만으로 판정하는 방식
- [ ] 기준표를 무엇으로 만들지 → `docs/reference_table_plan.md` 참고
