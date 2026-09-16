"""
=====================================================================
 lighting.py — [2단계] 흰 종이 기준 조명 보정 (화이트밸런스)
=====================================================================
 원본 : check2.py 의 white_balance()  ← PoC로 검증 완료 (물건색 차이 22.8 → 15.1)
        보정 '계산식'은 check2.py와 완전히 동일하게 유지했다.
        달라진 점은 (1) 재사용하기 쉽게 쪼갬 (2) 품질 진단 정보를 같이 돌려줌.

 원리 (한 줄 요약)
   "흰 종이는 원래 흰색(255,255,255)이어야 한다"
   → 사진 속 종이가 (200,220,240)처럼 찍혔다면 조명이 색을 입힌 것
   → 채널별로 255/200, 255/220, 255/240 배를 곱해 종이를 흰색으로 되돌리면
     같은 조명 아래 있던 얼굴·물건의 색도 함께 원래대로 돌아온다.
   (이론 이름: von Kries 방식 / "White Patch" 화이트밸런스)

 함수 구성 (위에서 아래로 쓰면 됨)
   estimate_gains(img, white_box)  종이 영역 → 채널별 배율(gain) 계산   ─┐
   apply_gains(img, gains)         배율을 사진 전체에 곱하기            ─┤ 둘을 합친 게
   white_balance(img, white_box)   위 두 개 + 품질 진단 → 결과 묶음 반환 ◀┘ 보통 이것만 쓰면 됨

 왜 estimate / apply 를 나눴나?
   웹캠(7단계)에서는 매 프레임마다 종이를 다시 찾을 필요 없이
   "처음 한 번 gain 계산 → 이후 프레임엔 apply만" 할 수 있어서.
   드레이핑(부가기능)에서도 같은 gain을 재사용할 수 있다.

 한계 (발표에도 명시)
   - 조명 색을 '줄여줄' 뿐 0으로 만들지 못한다 (카메라 자동보정·비선형 처리 때문)
   - 종이보다 밝은 곳(이마 반사광 등)은 255에서 잘린다 → 진단 정보로 경고
=====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from backend.color_utils import Box, clip_box, crop, mean_lab

# ---------------------------------------------------------------------
# 품질 경고 기준값 — 5단계(신뢰도)에서 알고리즘 파트가 튜닝할 숫자들.
# 코드 곳곳에 숫자를 박아두지 않고 여기 모아둔다.
# ---------------------------------------------------------------------
THRESHOLDS = {
    "paper_min_L": 40.0,          # 종이 밝기(L)가 이보다 어두우면: 배율이 커져 노이즈까지 증폭
    "max_gain": 3.0,              # 어떤 채널이든 3배 넘게 키워야 하면: 조명이 너무 어둡거나 치우침
    "paper_clip_value": 254,      # 종이 픽셀이 이 값 이상이면 센서 한계에 닿은(포화) 픽셀로 봄 → 진짜 조명색을 모름
    "paper_max_clip_ratio": 0.05, # 종이의 5% 이상이 포화면: 진짜 조명색을 알 수 없음
    "paper_max_std": 15.0,        # 종이 영역 밝기 표준편차가 크면: 그림자·글씨·줄무늬 섞였을 가능성
}


@dataclass
class WhiteBalanceResult:
    """
    white_balance()의 결과 묶음. (dict 대신 dataclass → 오타 시 에러로 바로 알 수 있음)
    result.image 처럼 점(.)으로 꺼내 쓴다.  dict가 필요하면 result.to_dict().
    """
    image: np.ndarray                 # 보정된 사진 (BGR, uint8) ← 다음 단계로 넘길 것
    gains: np.ndarray                 # 채널별 배율 [B, G, R]
    paper_bgr: np.ndarray             # 보정 전 종이 평균색 [B, G, R]
    paper_lab: np.ndarray             # 보정 전 종이 Lab → a,b가 0에서 멀수록 조명색이 강했다는 뜻
    cast_strength: float              # 조명 색끼 세기 = sqrt(a²+b²) of 종이 (0이면 원래 무채색 조명)
    paper_clip_ratio: float           # 종이 영역 중 포화(250+) 픽셀 비율
    paper_std: float                  # 종이 영역 밝기 표준편차 (균일한가?)
    newly_clipped_ratio: float        # 보정 때문에 새로 255에 잘린 픽셀 비율 (사진 전체)
    warnings: list[str] = field(default_factory=list)  # 사람이 읽을 경고 문구

    @property
    def ok(self) -> bool:
        """경고가 하나도 없으면 True"""
        return not self.warnings

    def to_dict(self) -> dict:
        """웹(JSON)으로 보낼 때 쓰는 형태. 이미지는 제외 (너무 큼)."""
        return {
            "gains_bgr": [round(float(g), 3) for g in self.gains],
            "paper_bgr": [round(float(v), 1) for v in self.paper_bgr],
            "paper_lab": [round(float(v), 1) for v in self.paper_lab],
            "cast_strength": round(self.cast_strength, 1),
            "paper_clip_ratio": round(self.paper_clip_ratio, 4),
            "paper_std": round(self.paper_std, 1),
            "newly_clipped_ratio": round(self.newly_clipped_ratio, 4),
            "warnings": list(self.warnings),
        }


def estimate_gains(img: np.ndarray, white_box: Box, target: float = 255.0) -> np.ndarray:
    """
    흰 종이 영역의 평균색으로 채널별 배율(gain)을 계산한다.

    img       : BGR 사진 (cv2.imread 결과 그대로)
    white_box : 흰 종이 영역 (x1, y1, x2, y2)
    target    : 종이를 몇으로 맞출지. 기본 255 = check2.py와 동일(검증된 값).
                (예: 240으로 주면 종이보다 밝은 부분이 덜 잘림 — 나중에 실험용)
    반환      : np.array([gainB, gainG, gainR])

    예) 종이 평균이 [200, 220, 240] → gain = [1.275, 1.159, 1.063]
    """
    avg = crop(img, white_box).astype(np.float32).mean(axis=(0, 1))  # [B, G, R] 평균
    avg[avg == 0] = 1.0            # 0으로 나누기 방지 (check2.py와 동일)
    return target / avg


def apply_gains(img: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """
    사진의 모든 픽셀에 채널별 배율을 곱한다. 255 넘는 값은 255로 자른다.
    reshape(1,1,3) : (세로, 가로, 3채널) 사진에 [B,G,R] 3개 배율을 한 번에 곱하기 위한 모양 맞춤
    """
    out = img.astype(np.float32) * np.asarray(gains, np.float32).reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def white_balance(img: np.ndarray, white_box: Box, target: float = 255.0) -> WhiteBalanceResult:
    """
    ★ 보통은 이 함수 하나만 쓰면 된다 ★
    흰 종이 기준으로 조명을 보정하고, 보정이 믿을 만한지 진단 정보까지 돌려준다.

    사용 예)
        import cv2
        from backend.vision.lighting import white_balance

        img = cv2.imread("data/samples/photo1.jpg")
        wb = white_balance(img, white_box=(645, 1817, 725, 1897))
        cv2.imwrite("outputs/corrected.jpg", wb.image)
        print(wb.gains, wb.warnings)
    """
    if img is None:
        raise ValueError("img 가 None 입니다. cv2.imread 경로를 확인하세요.")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"3채널 BGR 컬러 사진이 필요합니다. (받은 모양: {img.shape})")

    box = clip_box(white_box, img.shape)

    # ── 1) 보정 (check2.py와 동일한 계산) ─────────────────────────
    gains = estimate_gains(img, box, target)
    corrected = apply_gains(img, gains)

    # ── 2) 품질 진단 (check2.py엔 없던 부분) ──────────────────────
    paper = crop(img, box)
    paper_bgr = paper.reshape(-1, 3).astype(np.float32).mean(axis=0)
    paper_lab = mean_lab(img, box=box)
    cast_strength = float(np.hypot(paper_lab[1], paper_lab[2]))   # 종이의 a,b가 0에서 떨어진 정도

    t = THRESHOLDS
    paper_clip_ratio = float((paper >= t["paper_clip_value"]).any(axis=2).mean())
    paper_std = float(paper.astype(np.float32).mean(axis=2).std())  # 회색조로 본 밝기 편차

    # 보정 전엔 255 미만이었는데 보정 후 255가 된 픽셀 = 정보가 잘려 나간 픽셀
    was_ok = (img < 255).all(axis=2)
    now_clipped = (corrected == 255).any(axis=2)
    newly_clipped_ratio = float((was_ok & now_clipped).mean())

    warnings: list[str] = []
    if paper_lab[0] < t["paper_min_L"]:
        warnings.append(f"흰 종이가 너무 어둡습니다 (L={paper_lab[0]:.0f}). 조명을 밝게 해주세요.")
    if gains.max() > t["max_gain"]:
        warnings.append(f"보정 배율이 너무 큽니다 (최대 {gains.max():.1f}배). 조명이 어둡거나 색이 강합니다.")
    if paper_clip_ratio > t["paper_max_clip_ratio"]:
        warnings.append(f"흰 종이가 하얗게 날아갔습니다 ({paper_clip_ratio:.0%}). 노출을 낮추거나 빛 반사를 피하세요.")
    if paper_std > t["paper_max_std"]:
        warnings.append(f"흰 종이 영역이 균일하지 않습니다 (편차 {paper_std:.1f}). 그림자·글씨가 없는 곳을 지정하세요.")

    return WhiteBalanceResult(
        image=corrected,
        gains=gains,
        paper_bgr=paper_bgr,
        paper_lab=paper_lab,
        cast_strength=cast_strength,
        paper_clip_ratio=paper_clip_ratio,
        paper_std=paper_std,
        newly_clipped_ratio=newly_clipped_ratio,
        warnings=warnings,
    )
