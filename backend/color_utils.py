"""
=====================================================================
 color_utils.py — 모든 파트가 같이 쓰는 '색 계산' 공통 함수
=====================================================================
 왜 따로 뺐나?
   check2.py 안에 있던 measure_lab()은 조명 보정(2단계)뿐 아니라
   얼굴색 추출(3단계), 드레이핑 등 거의 모든 곳에서 쓰인다.
   한 곳에 두어야 "Lab 계산 방식"이 팀 전체에서 똑같이 유지된다.

 핵심 개념 — Lab 색 공간
   L : 밝기            0(검정) ~ 100(흰색)
   a : 초록(-) ~ 빨강(+)
   b : 파랑(-) ~ 노랑(+)   ← 웜/쿨 판정의 핵심 축 (b가 클수록 웜)
   RGB와 달리 '사람 눈이 느끼는 색 차이'와 거리가 비슷해서,
   두 색의 Lab 거리(ΔE)로 "얼마나 다른 색인가"를 말할 수 있다.

 주의 — OpenCV는 색 순서가 RGB가 아니라 BGR(파랑,초록,빨강)이다!
=====================================================================
"""
from __future__ import annotations

import cv2
import numpy as np

# 영역 좌표 형식: (x1, y1, x2, y2)  ← check2.py와 동일
#   x = 가로(왼→오), y = 세로(위→아래), x2/y2는 '포함하지 않는' 끝값
Box = tuple[int, int, int, int]


def clip_box(box: Box, img_shape) -> Box:
    """
    박스가 사진 밖으로 삐져나가면 사진 안쪽으로 잘라준다.
    (check2.py에는 없던 안전장치 — 좌표 실수로 빈 영역이 잡히는 걸 막음)
    """
    h, w = img_shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, x2 = sorted((max(0, min(w, x1)), max(0, min(w, x2))))
    y1, y2 = sorted((max(0, min(h, y1)), max(0, min(h, y2))))
    if x2 - x1 < 1 or y2 - y1 < 1:
        raise ValueError(f"영역 {box} 이(가) 사진({w}x{h}) 안에서 비어 있습니다. 좌표를 확인하세요.")
    return x1, y1, x2, y2


def crop(img: np.ndarray, box: Box) -> np.ndarray:
    """사진에서 박스 영역만 잘라낸다. (NumPy는 [세로, 가로] 순서로 자름에 주의)"""
    x1, y1, x2, y2 = clip_box(box, img.shape)
    return img[y1:y2, x1:x2]


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    BGR 사진(uint8, 0~255) → Lab 실수 배열 (L 0~100, a/b 약 -127~127)

    ※ check2.py와의 차이
       check2.py : uint8 그대로 변환 → L을 ×100/255, a·b를 -128 해서 되돌림
       여기      : 0~1 실수로 바꾼 뒤 변환 → OpenCV가 곧바로 '진짜 Lab 단위'를 줌
       결과는 소수점 수준(채널당 ±0.5 이내)으로 같고, 반올림 손실이 없어 더 정확하다.
       (tests/test_lighting.py 에서 원본 방식과의 차이를 자동 확인)
    """
    f = img_bgr.astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_BGR2LAB)


def mean_lab(img_bgr: np.ndarray, box: Box | None = None, mask: np.ndarray | None = None) -> np.ndarray:
    """
    영역의 평균색을 Lab [L, a, b] 로 반환.  (check2.py의 measure_lab 을 확장)

    영역 지정 방법 3가지:
      mean_lab(img)                → 사진 전체
      mean_lab(img, box=(x1,y1,x2,y2)) → 사각형 영역 (지금까지 쓰던 방식)
      mean_lab(img, mask=마스크)     → 모양이 자유로운 영역 (3단계: 볼·홍채·머리)
                                      mask는 사진과 같은 크기의 0/1(또는 0/255) 배열
    """
    if box is not None and mask is not None:
        raise ValueError("box 와 mask 중 하나만 지정하세요.")

    if box is not None:
        lab = bgr_to_lab(crop(img_bgr, box))
        return lab.reshape(-1, 3).mean(axis=0)

    lab = bgr_to_lab(img_bgr)
    if mask is None:
        return lab.reshape(-1, 3).mean(axis=0)

    sel = mask.astype(bool)
    if sel.shape != lab.shape[:2]:
        raise ValueError(f"mask 크기 {sel.shape} ≠ 사진 크기 {lab.shape[:2]}")
    if not sel.any():
        raise ValueError("mask 에 선택된 픽셀이 하나도 없습니다.")
    return lab[sel].mean(axis=0)


def delta_e(lab1, lab2) -> float:
    """
    두 Lab 색의 거리 ΔE (CIE76 = 단순 유클리드 거리). check2.py의 dist()와 같음.
    대략 기준:  ~2 사람 눈으로 거의 구분 불가 / ~10 확실히 다름 / 20+ 전혀 다른 색
    """
    return float(np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float)))
