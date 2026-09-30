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
    "value_steps": [150, 170, 190, 205, 220, 230, 240, 248],
    "_value_steps_설명": "밝기 하한을 점점 올려가며 종이를 찾는다. 흰 벽과 종이가 붙어 한 덩어리가 되면 "
                         "하한을 올려야 종이만 남는다 (종이가 보통 더 밝다)",
    "max_saturation": 60,   # 채도(S) 상한 — 이보다 색이 진하면 종이가 아님
    "open_kernel": 15,      # 작은 얼룩 제거 크기
    "center_ratio": 0.2,    # 덩어리 중심에서 (짧은 변 × 0.2) 만큼을 기준 영역으로
    "min_area_ratio": 0.01,  # 사진 전체의 1% 미만이면 종이로 인정하지 않음
    "max_area_ratio": 0.45,  # 45% 넘게 차지하면 벽·창문으로 본다
    "min_extent": 0.55,      # 덩어리가 자기 사각형을 이만큼은 채워야 한다 (종이는 네모)
    "max_border_sides": 2,   # 사진 테두리에 3면 이상 닿으면 벽으로 본다
    "prefer_area_ratio": 0.15,  # 이보다 커지면 점수를 깎는다 (손에 든 종이는 보통 사진의 2~15%)
    "border_penalty": 0.25,     # 테두리에 닿는 면 하나당 점수 25% 감점
    "min_clean_ratio": 0.5,     # 포화되지 않은 화소가 절반은 있어야 기준색으로 쓴다
}


TRIM = {
    "saturated": 250,      # 이 값 이상 = 하얗게 날아간 픽셀 → 기준에서 제외
    "shadow_percentile": 25,   # 어두운 쪽 25% (그림자·접힌 자국) 제외
    "min_pixels": 200,
}


class PaperNotFoundError(Exception):
    """흰 종이를 찾지 못했을 때. 웹에서는 '흰 종이가 보이게 다시 찍어주세요' 안내로 연결."""


def _find_component(img_bgr: np.ndarray):
    """흰 종이로 보이는 덩어리를 찾아 (라벨맵, 덩어리번호, stats, 중심) 을 돌려준다"""
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    k = PARAMS["open_kernel"]

    # 밝기 하한을 낮은 값부터 올려가며 '종이답게 생긴 덩어리' 후보를 모두 모은다.
    #   - 흰 벽이 있으면 낮은 하한에서는 벽+종이가 한 덩어리가 된다 → 하한을 올리면 종이만 남는다
    #   - 조건: 사진의 1~45% 크기, 네모난 모양, 사진 테두리에 3면 이상 닿지 않음
    #   - 조건을 만족하는 첫 덩어리에서 멈추지 않고 전부 점수를 매겨 가장 좋은 것을 고른다.
    #     (먼저 찾은 것에서 멈추면 JPEG 잡음 정도로도 벽이 뽑히는 일이 생긴다)
    best, best_score = None, -1.0
    blown_only = False      # '종이처럼 생긴 덩어리는 있었지만 전부 하얗게 날아간' 경우 구분용
    for v_min in PARAMS["value_steps"]:
        mask = ((hsv[:, :, 2] > v_min) & (hsv[:, :, 1] < PARAMS["max_saturation"])).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n):
            x, y, bw_, bh_, area = stats[i]
            ratio = area / (h * w)
            if not (PARAMS["min_area_ratio"] <= ratio <= PARAMS["max_area_ratio"]):
                continue
            extent = area / max(1, bw_ * bh_)
            if extent < PARAMS["min_extent"]:
                continue
            sides = (x <= 2) + (y <= 2) + (x + bw_ >= w - 2) + (y + bh_ >= h - 2)
            if sides > PARAMS["max_border_sides"]:
                continue
            sel = labels == i
            # 하얗게 날아간(포화) 화소가 많은 덩어리는 기준색으로 쓸 수 없다 → 바로 감점
            clean = float((img_bgr[sel] < TRIM["saturated"]).all(axis=1).mean())
            if clean < PARAMS["min_clean_ratio"]:
                blown_only = True
                continue
            bright = float(gray[sel].mean()) / 255.0          # 밝을수록 종이답다
            sat = float(hsv[:, :, 1][sel].mean())
            sat_ok = max(0.0, 1.0 - sat / PARAMS["max_saturation"])   # 색이 없을수록 종이답다
            size_ok = min(1.0, ratio / 0.03)                  # 너무 작으면 감점
            if ratio > PARAMS["prefer_area_ratio"]:           # 너무 크면 벽·창문 쪽으로 보고 감점
                size_ok *= PARAMS["prefer_area_ratio"] / ratio
            border_ok = 1.0 - PARAMS["border_penalty"] * sides   # 테두리에 닿을수록 감점
            score = clean * bright * extent * sat_ok * size_ok * max(0.0, border_ok)
            if score > best_score:
                best_score = score
                best = (labels, i, stats[i], centroids[i])

    if best is None:
        if blown_only:
            raise PaperNotFoundError(
                "흰 종이가 하얗게 날아가 기준색으로 쓸 수 없습니다. "
                "종이에 조명이 직접 닿지 않게 하고, 창이나 스탠드를 등지지 않은 자리에서 촬영해주세요.")
        raise PaperNotFoundError("흰 종이를 찾지 못했습니다. 종이를 얼굴 옆에 평평하게, 화면 안에 들어오게 들어주세요.")
    return best


def find_white_paper(img_bgr: np.ndarray) -> tuple[int, int, int, int]:
    """
    흰 종이의 '가운데 사각형' 좌표 (x1, y1, x2, y2). 사람이 손으로 찍던 것과 같은 형태.
    (지금은 white_reference() 를 주로 쓰고, 이 함수는 확인 도구·호환용으로 남겨둔다)
    """
    h, w = img_bgr.shape[:2]
    labels, idx, stats_i, centroid = _find_component(img_bgr)

    bw, bh = int(stats_i[cv2.CC_STAT_WIDTH]), int(stats_i[cv2.CC_STAT_HEIGHT])
    cx, cy = (int(round(v)) for v in centroid)
    s = max(5, int(min(bw, bh) * PARAMS["center_ratio"]))
    x1, y1 = max(0, cx - s), max(0, cy - s)
    x2, y2 = min(w, cx + s), min(h, cy + s)
    return x1, y1, x2, y2


def box_from_ratio(img_bgr: np.ndarray, ratio_box) -> tuple[int, int, int, int]:
    """
    웹에서 보내는 '비율 좌표'(0~1)를 픽셀 좌표로 바꾼다.
    웹 화면 크기와 실제 사진 크기가 달라도 맞게 변환된다.
      ratio_box: (x1, y1, x2, y2) 각각 0~1, 또는 dict
                 {"x1":..,"y1":..,"x2":..,"y2":..} / {"x":..,"y":..,"width":..,"height":..}
    """
    h, w = img_bgr.shape[:2]
    if isinstance(ratio_box, dict):
        d = ratio_box
        if "x1" in d:                                  # {x1,y1,x2,y2}
            ratio_box = (d["x1"], d["y1"], d["x2"], d["y2"])
        elif "width" in d or "w" in d:                 # {x,y,width,height} — 웹 프론트가 보내는 형식
            bw = d.get("width", d.get("w"))
            bh = d.get("height", d.get("h"))
            ratio_box = (d["x"], d["y"], d["x"] + bw, d["y"] + bh)
        else:
            raise ValueError(f"가이드 박스 좌표 형식을 알 수 없습니다: {sorted(d)}")
    x1, y1, x2, y2 = ratio_box
    if any(v is None or v != v for v in (x1, y1, x2, y2)):     # None·NaN
        raise ValueError("가이드 박스 좌표에 빈 값이 있습니다. 좌표 없이 자동 탐지로 진단해주세요.")
    # 이미 픽셀 좌표로 들어온 경우(1보다 큰 값)는 그대로 사용
    if max(x1, y1, x2, y2) > 1.5:
        return int(x1), int(y1), int(x2), int(y2)
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def white_reference(img_bgr: np.ndarray):
    """
    ★ 파이프라인이 실제로 쓰는 함수 ★
    흰 종이 덩어리 '전체'에서 아래 픽셀을 빼고 남은 것의 평균색을 기준으로 삼는다.
      - 하얗게 날아간 픽셀(250 이상)  → 진짜 색을 알 수 없음
      - 어두운 쪽 25%(그림자·접힌 자국) → 조명색이 아니라 그늘색

    사람이 종이에서 '깨끗한 부분'을 손으로 골라 찍던 것을, 자동으로 하는 것과 같다.
    반환: (기준색 BGR, 진단 dict)
      진단 dict 의 clean_ratio 가 낮으면(예: 0.4) 종이 절반이 날아간 것 → 신뢰도 낮춤
    """
    labels, idx, stats_i, centroid = _find_component(img_bgr)
    px = img_bgr[labels == idx].astype(np.float32)
    total = len(px)
    keep = (px < TRIM["saturated"]).all(axis=1)
    clean_ratio = float(keep.mean())
    if keep.sum() >= TRIM["min_pixels"]:
        px = px[keep]
    lum = px.mean(axis=1)
    lo = np.percentile(lum, TRIM["shadow_percentile"])
    sel = px[lum >= lo] if (lum >= lo).sum() >= TRIM["min_pixels"] else px

    x, y = int(stats_i[cv2.CC_STAT_LEFT]), int(stats_i[cv2.CC_STAT_TOP])
    bw, bh = int(stats_i[cv2.CC_STAT_WIDTH]), int(stats_i[cv2.CC_STAT_HEIGHT])
    info = {"clean_ratio": round(clean_ratio, 3), "pixels": int(total),
            "box": [x, y, x + bw, y + bh], "std": round(float(sel.mean(axis=1).std()), 2)}
    return sel.mean(axis=0), info
