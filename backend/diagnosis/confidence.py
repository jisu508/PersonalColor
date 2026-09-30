"""
=====================================================================
 confidence.py — [5단계] 신뢰도 0~100%
=====================================================================
 "이 진단을 얼마나 믿어도 되는가"를 숫자 하나로 보여준다.

 100점에서 시작해 문제가 있을 때마다 점수를 깎는다.
   ① 사진 품질  : 흐림, 너무 어둡거나 밝음, 얼굴이 작음, 옆얼굴
   ② 조명 보정  : 흰 종이 경고(어두움/포화/그림자), 보정 배율 과다
   ③ 결과 명확도: 1위와 2위 퍼센티지 격차가 작으면 애매한 것

 깎는 폭(감점표)은 아래 PENALTY 에 모아뒀다. 실제 사진으로 보정할 값이다.
=====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

PENALTY = {
    "blurry": 25,            # 흐린 사진
    "too_dark": 20,
    "too_bright": 20,
    "small_face": 15,
    "side_face": 15,
    "lighting_warning": 10,  # 조명 보정 경고 1건당 (최대 30)
    "lighting_warning_max": 30,
    "face_warning": 5,       # 얼굴 추출 경고 1건당 (최대 15)
    "face_warning_max": 15,
    "gate_warning": 12,        # 촬영 조건 경고 하나당 (막지는 않은 것)
    "gate_warning_max": 30,
    "hair_missing": 5,
    "eye_missing": 5,
    "unclear_10": 20,        # 1·2위 격차 10%p 미만
    "unclear_25": 10,        # 1·2위 격차 25%p 미만
}

THRESHOLDS = {"min_sharpness": 10.0, "min_brightness": 35.0, "max_brightness": 88.0,
              "min_face_width_ratio": 0.15, "max_asymmetry": 0.15}


@dataclass
class ConfidenceResult:
    score: int                  # 0~100
    message: str                # 화면에 보여줄 한 줄 (예: "조명 충분")
    reasons: list = field(default_factory=list)   # 깎인 이유 목록

    def to_dict(self) -> dict:
        return {"score": self.score, "message": self.message, "reasons": list(self.reasons)}


def compute_confidence(quality: dict, lighting: dict | None = None,
                       season_gap: float | None = None, colors: dict | None = None,
                       gate_warnings: list[str] | None = None) -> ConfidenceResult:
    """
    quality    : face_color 결과의 quality 부분 (brightness, sharpness, face_width_ratio, asymmetry)
    lighting   : white_balance 결과의 to_dict() (warnings 등). 없으면 생략
    season_gap : 1위 - 2위 퍼센티지 차이
    colors     : {"skin":..., "eye":..., "hair":...} — 빠진 부위가 있으면 조금 감점
    gate_warnings : 촬영 조건 경고(막지는 않은 것들). 하나당 감점
    """
    score = 100
    reasons = []
    t = THRESHOLDS

    q = quality or {}
    if q.get("sharpness", 999) < t["min_sharpness"]:
        score -= PENALTY["blurry"]; reasons.append("사진이 흐림")
    if q.get("brightness", 50) < t["min_brightness"]:
        score -= PENALTY["too_dark"]; reasons.append("사진이 어두움")
    elif q.get("brightness", 50) > t["max_brightness"]:
        score -= PENALTY["too_bright"]; reasons.append("사진이 너무 밝음")
    if q.get("face_width_ratio", 1) < t["min_face_width_ratio"]:
        score -= PENALTY["small_face"]; reasons.append("얼굴이 작게 찍힘")
    if q.get("asymmetry", 0) > t["max_asymmetry"]:
        score -= PENALTY["side_face"]; reasons.append("얼굴이 옆으로 돌아감")

    if lighting:
        n = len(lighting.get("warnings", []))
        if n:
            cut = min(n * PENALTY["lighting_warning"], PENALTY["lighting_warning_max"])
            score -= cut; reasons.append(f"조명 보정 경고 {n}건")

    if gate_warnings:
        cut = min(len(gate_warnings) * PENALTY["gate_warning"], PENALTY["gate_warning_max"])
        score -= cut; reasons.append(f"촬영 조건 경고 {len(gate_warnings)}건")

    if colors:
        if colors.get("hair") is None:
            score -= PENALTY["hair_missing"]; reasons.append("머리카락 색 추출 실패")
        if colors.get("eye") is None:
            score -= PENALTY["eye_missing"]; reasons.append("눈동자 색 추출 실패")

    if season_gap is not None:
        if season_gap < 10:
            score -= PENALTY["unclear_10"]; reasons.append("1·2위 시즌 차이가 거의 없음")
        elif season_gap < 25:
            score -= PENALTY["unclear_25"]; reasons.append("두 시즌 경계에 가까움")

    score = int(max(0, min(100, score)))
    if score >= 85:
        msg = "조명 충분"
    elif score >= 70:
        msg = "참고용으로 충분"
    elif score >= 50:
        msg = "조건이 좋지 않음"
    else:
        msg = "재촬영 권장"
    return ConfidenceResult(score=score, message=msg, reasons=reasons)
