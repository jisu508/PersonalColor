"""
[6단계 — 아직 미구현] 전체 파이프라인: 사진 1장 → 결과 dict

    사진 ─▶ vision.lighting.white_balance ─▶ vision.face_color.extract_face_colors
         ─▶ diagnosis.season.diagnose_season ─▶ diagnosis.confidence.compute_confidence
         ─▶ {"seasons": {...}, "confidence": 0.92, "colors": {...}, "lighting": {...}}

웹(web/app.py)은 이 파일의 run_pipeline() 하나만 호출한다.
"""


def run_pipeline(img_bgr, white_box):
    raise NotImplementedError("6단계에서 구현 예정")
