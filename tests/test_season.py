"""
tests/test_season.py — 4단계(시즌 판정)·5단계(신뢰도)·6단계(파이프라인) 테스트
실행: python -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.diagnosis.confidence import compute_confidence                      # noqa: E402
from backend.diagnosis.season import (compare_methods, diagnose_season,          # noqa: E402
                                      load_reference, warm_cool_score)

WARM = {"skin": [63.45, 11.80, 17.11], "hair": [25.0, 3.0, 5.0], "eye": [20.0, 5.0, 3.0]}   # 논문① 웜 그룹 평균
COOL = {"skin": [64.80, 10.87, 15.04], "hair": [25.0, 3.0, 5.0], "eye": [20.0, 5.0, 3.0]}   # 논문① 쿨 그룹 평균


# ── 판정식 자체 ──────────────────────────────────────────
def test_paper_d_matches_published_centroids():
    """논문①의 웜·쿨 그룹 평균을 넣으면 논문에 적힌 중심점(1.338 / -0.412)이 나와야 한다"""
    ref = load_reference()
    c = ref["warm_cool"]["paper_d"]["coef"]
    D = lambda lab: c["const"] + c["a"] * lab[1] + c["b"] * lab[2]   # noqa: E731
    assert abs(D(WARM["skin"]) - 1.338) < 0.05
    assert abs(D(COOL["skin"]) - (-0.412)) < 0.05


def test_paper_d_separates_paper_groups():
    """논문 기준값(0.463)으로 두 그룹이 갈려야 한다"""
    assert warm_cool_score(WARM, method="paper_d")[0] > 0
    assert warm_cool_score(COOL, method="paper_d")[0] < 0


def test_b_method_uses_threshold():
    ref = load_reference()
    t = ref["warm_cool"]["b"]["threshold"]
    assert warm_cool_score({"skin": [65, 12, t + 3]}, method="b")[0] > 0
    assert warm_cool_score({"skin": [65, 12, t - 3]}, method="b")[0] < 0


def test_b_method_ignores_redness():
    """홍조(a)가 달라져도 b 방식 결과는 같아야 한다"""
    s1 = warm_cool_score({"skin": [65, 8, 18]}, method="b")[0]
    s2 = warm_cool_score({"skin": [65, 25, 18]}, method="b")[0]
    assert s1 == s2


def test_smtc_needs_at_least_one_part():
    with pytest.raises(ValueError):
        warm_cool_score({"skin": None, "hair": None, "eye": None}, method="smtc")


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        warm_cool_score(WARM, method="없는방법")


# ── 퍼센티지 ────────────────────────────────────────────
def test_percentages_sum_to_100():
    r = diagnose_season(WARM)
    assert abs(sum(r.seasons.values()) - 100) < 0.01
    assert len(r.seasons) == 4


def test_warm_skin_gets_warm_season_on_top():
    r = diagnose_season({"skin": [65, 12, 24]})           # b가 아주 높음 → 웜
    assert r.warm_cool == "warm" and r.top.endswith("warm")


def test_cool_skin_gets_cool_season_on_top():
    r = diagnose_season({"skin": [65, 12, 8]})            # b가 아주 낮음 → 쿨
    assert r.warm_cool == "cool" and r.top.endswith("cool")


def test_borderline_has_small_gap():
    """경계에 있는 사람은 1·2위 격차가 작아야 한다(= 퍼센티지로 보여줄 값)"""
    ref = load_reference()
    t = ref["warm_cool"]["b"]["threshold"]
    near = diagnose_season({"skin": [65, 12, t]}, method="b").gap
    far = diagnose_season({"skin": [65, 12, t + 8]}, method="b").gap
    assert near < far


def test_top_season_matches_warm_percent():
    """웜 확률이 50%가 넘으면 1위 시즌도 웜 쪽이어야 한다 (앞뒤가 맞아야 함)"""
    for skin in ([65, 25.6, 23.5], [65, 10.4, 14.1], [65, 13.9, 17.1], [70, 8, 25], [70, 20, 8]):
        r = diagnose_season({"skin": skin})
        side = "warm" if r.warm_percent >= 50 else "cool"
        assert r.top.endswith(side), (skin, r.top, r.warm_percent)


def test_compare_methods_returns_all_methods():
    out = compare_methods(WARM)
    assert set(out) == {"bc", "hue", "b", "paper_d", "smtc"}


def test_hue_matches_webcam_labels():
    """라벨이 확실한 웹캠 3장(현재 파이프라인 측정값): 웜 51.2°, 50.0° / 쿨 43.1°"""
    import numpy as np
    for h_deg, expect in ((51.2, "warm"), (50.0, "warm"), (43.1, "cool"), (33.0, "cool")):
        a = 10.0
        b = a * np.tan(np.radians(h_deg))
        r = diagnose_season({"skin": [65, a, b]})
        assert r.warm_cool == expect, (h_deg, r.warm_cool, r.warm_percent)


def test_borderline_band_marks_middle():
    """기준선 부근(±band)은 웜/쿨을 단정하지 않고 '경계형' 으로 표시한다"""
    import numpy as np
    ref = load_reference()["warm_cool"]["hue"]
    a = 10.0
    b = a * np.tan(np.radians(ref["threshold"]))
    assert diagnose_season({"skin": [65, a, b]}).warm_cool == "borderline"


def test_bc_method_runs_and_is_negative():
    """b - C 방법은 계산은 되지만 항상 0 이하 값이 나온다 (비교용으로만 남겨둠)"""
    import numpy as np
    for a, b in [(5, 20), (20, 5), (10, 10)]:
        score, detail = warm_cool_score({"skin": [70, a, b]}, method="bc")
        assert detail["D_bc"] <= 0


def test_default_method_is_hue():  # noqa: D401
    """현재 기본 판정 방법은 색상각 hue (팀 결정)"""
    assert load_reference()["warm_cool"]["method"] == "hue"


def test_missing_skin_raises():
    with pytest.raises(ValueError):
        diagnose_season({"skin": None})


# ── 신뢰도 ──────────────────────────────────────────────
GOOD_Q = {"brightness": 65.0, "sharpness": 40.0, "face_width_ratio": 0.35, "asymmetry": 0.05}


def test_confidence_good_photo_is_high():
    c = compute_confidence(GOOD_Q, lighting={"warnings": []}, season_gap=50,
                           colors={"skin": [1, 2, 3], "hair": [1, 2, 3], "eye": [1, 2, 3]})
    assert c.score >= 85 and c.reasons == []


def test_confidence_blurry_photo_drops():
    bad = {**GOOD_Q, "sharpness": 3.0}
    assert compute_confidence(bad, season_gap=50).score < compute_confidence(GOOD_Q, season_gap=50).score


def test_confidence_borderline_result_drops():
    assert compute_confidence(GOOD_Q, season_gap=5).score < compute_confidence(GOOD_Q, season_gap=50).score


def test_confidence_stays_in_range():
    worst = {"brightness": 10, "sharpness": 0.1, "face_width_ratio": 0.02, "asymmetry": 0.9}
    c = compute_confidence(worst, lighting={"warnings": ["a", "b", "c", "d"]}, season_gap=1,
                           colors={"skin": [1, 2, 3], "hair": None, "eye": None})
    assert 0 <= c.score <= 100


# ── 파이프라인 (사진 있을 때만) ──────────────────────────
SAMPLE = ROOT / "data/samples/photo1.jpg"


@pytest.mark.skipif(not SAMPLE.is_file(), reason="data/samples/photo1.jpg 없음")
def test_pipeline_end_to_end():
    pytest.importorskip("mediapipe")
    from backend.image_io import imread
    from backend.pipeline import run_pipeline
    r = run_pipeline(imread(SAMPLE), guide_box=(645, 1817, 725, 1897))
    assert r["ok"] is True
    assert abs(sum(p["value"] for p in r["percentages"]) - 100) < 0.5
    assert 0 <= r["reliability"] <= 100
    assert r["colors"]["skin"] is not None


def test_pipeline_rejects_non_face_image():
    from backend.pipeline import run_pipeline
    r = run_pipeline(np.full((400, 400, 3), 128, np.uint8))
    assert r["ok"] is False and "message" in r
