"""
=====================================================================
 paper.py — 사진에서 '흰 종이'를 자동으로 찾기
=====================================================================
 왜 필요? 지금까지는 흰 종이 좌표를 손으로 넣었는데, 웹에서 실시간 촬영할 때는
          사람이 좌표를 찍어줄 수 없다. 그래서 자동으로 찾는다.

 찾는 방법 (아주 단순)
   1) 밝고(밝기 높음) 색이 거의 없는(채도 낮음) 픽셀만 남긴다   ← 흰 종이의 특징
   2) 작은 얼룩은 지운다
   3) 그중 '가장 큰 덩어리'를 흰 종이로 본다
   4) 그 덩어리의 한가운데 정사각형을 기준 영역으로 쓴다
      (가장자리는 그림자·접힌 자국이 있을 수 있어서 피한다)

 웹에서 가이드 박스를 쓰는 경우엔 그 좌표를 그대로 넘기면 되고,
 좌표가 없을 때만 이 함수가 대신 찾아준다.
=====================================================================
"""
from __future__ import annotations

import cv2
import numpy as np

PARAMS = {
    "min_value": 150,      # 밝기(V) 하한 — 이보다 어두우면 종이로 안 봄
    "max_saturation": 60,  # 채도(S) 상한 — 이보다 색이 진하면 종이가 아님
    "open_kernel": 15,     # 작은 얼룩 제거 크기
    "center_ratio": 0.2,   # 덩어리 중심에서 (짧은 변 × 0.2) 만큼을 기준 영역으로
    "min_area_ratio": 0.01,  # 사진 전체의 1% 미만이면 종이로 인정하지 않음
}


class PaperNotFoundError(Exception):
    """흰 종이를 찾지 못했을 때. 웹에서는 '흰 종이가 보이게 다시 찍어주세요' 안내로 연결."""


def find_white_paper(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    사진에서 흰 종이를 찾아 기준 영역 (x1, y1, x2, y2) 을 돌려준다.
    못 찾으면 PaperNotFoundError.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 2] > PARAMS["min_value"]) & (hsv[:, :, 1] < PARAMS["max_saturation"])).astype(np.uint8)
    k = PARAMS["open_kernel"]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        raise PaperNotFoundError("사진에서 흰 종이를 찾지 못했습니다. 흰 종이가 화면에 보이게 촬영해주세요.")

    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[idx, cv2.CC_STAT_AREA]
    if area / (h * w) < PARAMS["min_area_ratio"]:
        raise PaperNotFoundError("흰 종이가 너무 작게 찍혔습니다. 종이를 얼굴 옆에 더 크게 들어주세요.")

    bw, bh = stats[idx, cv2.CC_STAT_WIDTH], stats[idx, cv2.CC_STAT_HEIGHT]
    cx, cy = (int(round(v)) for v in centroids[idx])
    s = max(5, int(min(bw, bh) * PARAMS["center_ratio"]))
    x1, y1 = max(0, cx - s), max(0, cy - s)
    x2, y2 = min(w, cx + s), min(h, cy + s)
    return x1, y1, x2, y2


def box_from_ratio(img_bgr: np.ndarray, ratio_box) -> tuple[int, int, int, int]:
    """
    웹에서 보내는 '비율 좌표'(0~1)를 픽셀 좌표로 바꾼다.
    웹 화면 크기와 실제 사진 크기가 달라도 맞게 변환된다.
      ratio_box: (x1, y1, x2, y2) 각각 0~1 또는 {"x1":..,"y1":..,"x2":..,"y2":..}
    """
    h, w = img_bgr.shape[:2]
    if isinstance(ratio_box, dict):
        ratio_box = (ratio_box["x1"], ratio_box["y1"], ratio_box["x2"], ratio_box["y2"])
    x1, y1, x2, y2 = ratio_box
    # 이미 픽셀 좌표로 들어온 경우(1보다 큰 값)는 그대로 사용
    if max(x1, y1, x2, y2) > 1.5:
        return int(x1), int(y1), int(x2), int(y2)
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
