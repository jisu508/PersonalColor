"""
tests/test_lighting.py — 조명 보정 자동 테스트  (담당: 문서·테스트)

실행 (프로젝트 최상위 폴더에서):   python -m pytest tests -v

실제 사진 없이 '가짜 사진'을 만들어 테스트한다:
  1) 진짜 색을 아는 장면(흰 종이 + 색 있는 물건)을 만든다
  2) 노란 조명을 씌운 것처럼 채널별로 색을 곱해 망가뜨린다
  3) white_balance 로 되돌렸을 때 원래 색에 가까워지는지 확인
→ 누가 lighting.py 를 고치다 망가뜨리면 이 테스트가 바로 잡아낸다.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.color_utils import delta_e, mean_lab            # noqa: E402
from backend.vision.lighting import apply_gains, estimate_gains, white_balance  # noqa: E402

PAPER = (0, 0, 100, 100)     # 가짜 사진에서 흰 종이 위치
OBJ = (150, 0, 250, 100)     # 가짜 사진에서 물건 위치


def make_scene(illum_bgr=(1.0, 1.0, 1.0), paper_value=230):
    """흰 종이(왼쪽) + 살구색 물건(오른쪽) 장면. illum_bgr = 조명이 채널별로 곱하는 값."""
    img = np.zeros((100, 250, 3), np.float32)
    img[:, 0:100] = paper_value                 # 종이 (무채색)
    img[:, 150:250] = (140, 170, 220)           # 물건 BGR (살구색)
    img[:, 100:150] = (60, 60, 60)              # 사이 배경
    img *= np.array(illum_bgr, np.float32).reshape(1, 1, 3)
    return np.clip(img, 0, 255).astype(np.uint8)


def legacy_white_balance(img, white_box):
    """check2.py 원본 코드 그대로 (비교 기준)"""
    x1, y1, x2, y2 = white_box
    avg = img[y1:y2, x1:x2].astype(np.float32).mean(axis=(0, 1))
    avg[avg == 0] = 1.0
    out = img.astype(np.float32) * (255.0 / avg).reshape(1, 1, 3)
    return np.clip(out, 0, 255).astype(np.uint8)


def legacy_measure_lab(img, box):
    """check2.py 원본 measure_lab 그대로"""
    x1, y1, x2, y2 = box
    lab = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.array([lab[:, :, 0].mean() * 100 / 255, lab[:, :, 1].mean() - 128, lab[:, :, 2].mean() - 128])


def test_same_as_check2():
    """새 함수의 보정 결과가 검증된 check2.py 와 픽셀 단위로 완전히 같아야 한다."""
    img = make_scene(illum_bgr=(0.7, 0.9, 1.05))
    assert np.array_equal(white_balance(img, PAPER).image, legacy_white_balance(img, PAPER))


def test_lab_matches_check2_within_half():
    """Lab 계산 방식 변경(uint8→float)의 차이는 ΔE 1.0 미만이어야 한다.
    (check2 방식은 a·b를 정수로 반올림하므로 단색 영역에선 채널당 최대 ±0.5 차이가 난다.
     실제 사진처럼 픽셀 색이 다양하면 평균에서 반올림 오차가 상쇄돼 훨씬 작아진다.)"""
    img = make_scene(illum_bgr=(0.7, 0.9, 1.05))
    assert delta_e(mean_lab(img, box=OBJ), legacy_measure_lab(img, OBJ)) < 1.0


def test_paper_becomes_white():
    img = make_scene(illum_bgr=(0.6, 0.85, 1.0))
    wb = white_balance(img, PAPER)
    L, a, b = mean_lab(wb.image, box=PAPER)
    assert L > 99 and abs(a) < 1 and abs(b) < 1


def test_different_lights_converge():
    """핵심 가설: 조명이 달라도 보정 후엔 물건색 차이가 줄어든다."""
    yellow = make_scene(illum_bgr=(0.55, 0.85, 1.0))   # 노란 조명
    blue = make_scene(illum_bgr=(1.0, 0.95, 0.8))      # 푸른 조명
    before = delta_e(mean_lab(yellow, box=OBJ), mean_lab(blue, box=OBJ))
    after = delta_e(mean_lab(white_balance(yellow, PAPER).image, box=OBJ),
                    mean_lab(white_balance(blue, PAPER).image, box=OBJ))
    assert after < before * 0.2


def test_estimate_apply_equals_white_balance():
    img = make_scene(illum_bgr=(0.8, 0.9, 1.0))
    assert np.array_equal(apply_gains(img, estimate_gains(img, PAPER)), white_balance(img, PAPER).image)


def test_warning_dark_paper():
    wb = white_balance(make_scene(paper_value=40), PAPER)
    assert not wb.ok and any("어둡" in w for w in wb.warnings)


def test_warning_saturated_paper():
    wb = white_balance(make_scene(paper_value=255), PAPER)
    assert any("날아갔" in w for w in wb.warnings)


def test_good_scene_has_no_warning():
    wb = white_balance(make_scene(illum_bgr=(0.8, 0.9, 1.0)), PAPER)
    assert wb.ok, wb.warnings


def test_box_outside_image_raises():
    with pytest.raises(ValueError):
        white_balance(make_scene(), (500, 500, 600, 600))


def test_to_dict_is_json_serializable():
    import json
    json.dumps(white_balance(make_scene(), PAPER).to_dict(), ensure_ascii=False)
