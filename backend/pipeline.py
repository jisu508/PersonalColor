"""
=====================================================================
 pipeline.py — [6단계] 사진 한 장 → 최종 결과
=====================================================================
 웹(app.py)은 이 파일의 run_pipeline() 하나만 부르면 된다.

   사진 ─▶ ① 흰 종이 찾기 ─▶ ② 조명 보정 ─▶ ③ 얼굴 부위별 색 추출
        ─▶ ④ 시즌 퍼센티지 ─▶ ⑤ 신뢰도 ─▶ 결과 dict

 사용 예 (Flask):
     from backend.image_io import imdecode_bytes
     from backend.pipeline import run_pipeline
     img = imdecode_bytes(file.read())
     result = run_pipeline(img, guide_box=coords)   # coords 는 웹에서 보낸 가이드 박스
     return jsonify(result)
=====================================================================
"""
from __future__ import annotations

import numpy as np

import numpy as np

from backend.color_utils import delta_e
from backend.diagnosis.confidence import compute_confidence
from backend.diagnosis.season import diagnose_season, load_reference, season_color, season_label
from backend.vision.face_color import extract_face_colors
from backend.vision.landmarks import FaceNotFoundError
from backend.vision.lighting import white_balance
from backend.vision.paper import PaperNotFoundError, box_from_ratio, white_reference


def run_pipeline(img_bgr: np.ndarray, guide_box=None, method: str | None = None,
                 skip_white_balance: bool = False, return_images: bool = False) -> dict:
    """
    사진 한 장을 넣으면 진단 결과 dict 를 돌려준다.

    img_bgr   : BGR 사진
    guide_box : 웹 가이드 박스 좌표. (x1,y1,x2,y2) 픽셀 또는 0~1 비율, dict 도 가능.
                None 이면 사진에서 흰 종이를 자동으로 찾는다.
    method    : 웜/쿨 판정 방법 ("b" / "paper_d" / "smtc"). None 이면 JSON 기본값
    skip_white_balance : True 면 조명 보정 없이 진단 (수집 사진 실험용)
    return_images : True 면 결과에 images={"corrected": 보정된 BGR 배열} 을 넣는다.
                    (JSON 으로 못 보내는 값이라 app.py 가 꺼내서 base64 로 바꾼다)

    실패해도 예외를 던지지 않고 {"ok": False, "error": ..., "message": 안내문} 형태로 돌려준다.
    """
    ref = load_reference()
    lighting_info = None
    warnings: list[str] = []

    # ── ①②  흰 종이 → 조명 보정 ─────────────────────────────
    work = img_bgr
    if not skip_white_balance:
        try:
            target = ref.get("white_balance", {}).get("target", 255.0)
            if guide_box is not None:
                box = box_from_ratio(img_bgr, guide_box)
                wb = white_balance(img_bgr, box, target=target)
                paper_info = {"mode": "guide_box", "box": list(box)}
            else:
                # 종이 전체에서 포화·그림자를 뺀 평균색을 기준으로 (사람이 깨끗한 곳 찍던 것과 같은 효과)
                ref_bgr, paper_info = white_reference(img_bgr)
                paper_info["mode"] = "auto"
                wb = white_balance(img_bgr, target=target, reference_bgr=ref_bgr)
            work = wb.image
            lighting_info = wb.to_dict()
            lighting_info["paper"] = paper_info
            warnings += wb.warnings
        except PaperNotFoundError as e:
            return {"ok": False, "error": "paper_not_found", "message": str(e)}
        except ValueError as e:
            return {"ok": False, "error": "white_balance_failed", "message": str(e)}

    # ── ③  얼굴 부위별 색 ────────────────────────────────────
    # 보정본으로 먼저 시도하고, 실패하면 보정 없이 한 번 더 시도한다.
    # (자동으로 찾은 '흰 종이'가 사실은 밝은 배경이어서 사진이 망가지는 경우 대비)
    fc = None
    try:
        fc = extract_face_colors(work)
        if fc.skin is None:
            raise ValueError("피부색 추출 실패")
    except (FaceNotFoundError, ValueError) as first_error:
        if work is img_bgr:
            return {"ok": False, "error": "face_not_found", "message": str(first_error)}
        try:
            fc = extract_face_colors(img_bgr)          # 보정 없이 재시도
            if fc.skin is None:
                raise ValueError("피부색 추출 실패")
            warnings.append("조명 보정 결과로는 얼굴색을 읽지 못해, 보정 없이 진단했습니다. "
                            "흰 종이가 화면에 잘 보이게 다시 촬영해주세요.")
            lighting_info = None
        except (FaceNotFoundError, ValueError) as e:
            return {"ok": False, "error": "face_not_found", "message": str(e)}

    face = fc.to_dict()
    warnings += fc.warnings
    colors = {"skin": face["skin"], "eye": face["eye"], "hair": face["hair"]}

    # ── ③-1  촬영 조건 점검 (못 맞추면 판정 대신 재촬영 안내) ─────
    gate = ref.get("quality_gate", {})
    reasons = []
    if gate.get("require_white_balance") and lighting_info is None and not skip_white_balance:
        reasons.append("흰 종이 조명 보정을 하지 못했습니다. 흰 종이가 화면에 잘 보이게 촬영해주세요.")
    cl, cr = face["regions"]["cheek_left"]["lab"], face["regions"]["cheek_right"]["lab"]
    cheek_de = delta_e(cl, cr) if (cl and cr) else None
    if cheek_de is not None and cheek_de > gate.get("max_cheek_delta_e", 99):
        reasons.append(f"얼굴 좌우 밝기·색 차이가 큽니다 (ΔE {cheek_de:.1f}). 조명을 정면으로 받는 자리에서 촬영해주세요.")
    if lighting_info:
        paper = lighting_info.get("paper", {})
        blown = 1.0 - paper["clean_ratio"] if "clean_ratio" in paper else lighting_info["paper_clip_ratio"]
        if blown > gate.get("max_paper_clip_ratio", 1.0):
            reasons.append(f"흰 종이가 하얗게 날아갔습니다 ({blown:.0%}). 종이에 조명이 직접 닿지 않게 해주세요.")
    if face["quality"]["face_width_ratio"] < gate.get("min_face_width_ratio", 0):
        reasons.append(f"얼굴이 작게 찍혔습니다 (사진 폭의 {face['quality']['face_width_ratio']:.0%}). 조금 더 가까이서 촬영해주세요.")

    # ── ④  시즌 퍼센티지 ─────────────────────────────────────
    season = diagnose_season(colors, ref=ref, method=method)

    # ── ⑤  신뢰도 ────────────────────────────────────────────
    conf = compute_confidence(quality=face["quality"], lighting=lighting_info,
                              season_gap=season.gap, colors=colors)

    # ── 결과 정리 (웹 화면이 바로 쓰는 형태 + 디버그용 원본 숫자) ──
    percentages = [{"key": k, "name": season_label(k, ref), "value": round(v, 1), "color": season_color(k, ref)}
                   for k, v in sorted(season.seasons.items(), key=lambda kv: -kv[1])]

    warm_pct = round(season.warm_percent, 1)
    tone = {"warm": round(warm_pct, 1), "cool": round(100 - warm_pct, 1),
            "verdict": season.warm_cool,
            "label": {"warm": "웜", "cool": "쿨", "borderline": "경계형"}[season.warm_cool]}

    images = {"corrected": work} if return_images else None

    if reasons:   # 촬영 조건 미달 → 판정하지 않고 재촬영 안내
        return {"ok": False, "error": "retake_required", "message": reasons[0], "reasons": reasons,
                "images": images,
                "debug": {"tone": tone, "percentages": percentages, "colors": colors,
                          "quality": face["quality"], "lighting": lighting_info,
                          "cheek_delta_e": round(cheek_de, 1) if cheek_de is not None else None}}

    return {
        "ok": True,
        "tone": tone,
        "images": images,
        # 화면 표시용
        "percentages": percentages,
        "best_group": season_label(season.top, ref),
        "reliability": conf.score,
        "reliability_msg": conf.message,
        "warnings": warnings,
        "message": _interpret(season),
        # 코드에서 쓰는 값
        "seasons": season.to_dict()["seasons"],
        "top": season.top,
        "warm_cool": season.warm_cool,
        "confidence": conf.to_dict(),
        "colors": colors,
        "quality": face["quality"],
        "lighting": lighting_info,
        "season_detail": season.to_dict()["detail"],
        "cheek_delta_e": round(cheek_de, 1) if cheek_de is not None else None,
        "method": season.method,
    }


def _interpret(season) -> str:
    """경계 케이스 해석 문구 (개발계획서 부가기능 ⑥)"""
    if season.warm_cool == "borderline":
        return (f"웜 {season.warm_percent:.0f}% / 쿨 {100 - season.warm_percent:.0f}% — 경계형입니다. "
                "양쪽 색을 모두 활용할 수 있습니다.")
    if season.gap >= 40:
        from backend.diagnosis.season import season_label
        return f"전형적인 {season_label(season.top)} 유형입니다."
    if season.gap >= 20:
        return "한쪽 시즌이 우세하지만 다른 시즌 요소도 있습니다."
    return "두 시즌 경계에 있어 양쪽 색을 모두 활용할 수 있습니다."
