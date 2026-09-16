"""
tests/test_face_color.py — 3단계 얼굴색 추출 테스트

  [A] MediaPipe 없이 도는 테스트 : 픽셀 걸러내기·마스크 같은 계산 부품 검사
  [B] 실제 사진 테스트          : data/samples/photo1.jpg, photo3.jpg 가 있을 때만 실행
                                  (사진은 GitHub에 없으므로, 없으면 자동으로 SKIPPED 표시)
실행: python -m pytest tests -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.color_utils import bgr_pixels_to_lab, delta_e, mean_lab   # noqa: E402
from backend.vision.face_color import (pixels_in_mask, polygon_mask,   # noqa: E402
                                       ring_mask, robust_mean_lab)

SKIN_BGR = (150, 170, 215)   # 살구색 피부 (가짜)


# ───────────── [A] 계산 부품 ─────────────
def test_pixel_lab_same_as_image_lab():
    """픽셀 목록 변환(bgr_pixels_to_lab)이 사진 전체 변환(mean_lab)과 같은 값인지"""
    img = np.full((10, 10, 3), SKIN_BGR, np.uint8)
    assert delta_e(bgr_pixels_to_lab(img.reshape(-1, 3)).mean(axis=0), mean_lab(img)) < 1e-3


def test_robust_mean_ignores_highlight_and_shadow():
    """피부 픽셀에 반사광(흰색)·그림자(검정)가 섞여도 평균이 피부색에 가까워야 한다"""
    skin = np.tile(np.array(SKIN_BGR, np.uint8), (800, 1))
    skin = skin + np.random.default_rng(0).integers(-6, 7, skin.shape)          # 자연스러운 노이즈
    highlight = np.full((80, 3), 255, np.int64)
    shadow = np.full((80, 3), 20, np.int64)
    px = np.clip(np.vstack([skin, highlight, shadow]), 0, 255).astype(np.uint8)

    true_lab = bgr_pixels_to_lab(np.array([SKIN_BGR], np.uint8))[0]
    naive = bgr_pixels_to_lab(px).mean(axis=0)
    robust, n = robust_mean_lab(px)
    assert delta_e(robust, true_lab) < 2.0              # 걸러낸 평균은 진짜 피부색과 거의 같고
    assert delta_e(naive, true_lab) > delta_e(robust, true_lab) * 3   # 그냥 평균보다 훨씬 정확
    assert n < len(px)


def test_robust_mean_empty():
    assert robust_mean_lab(np.empty((0, 3), np.uint8)) == (None, 0)


def test_polygon_mask_area():
    """정사각형 다각형 마스크의 픽셀 수가 넓이와 맞는지"""
    m = polygon_mask((100, 100), np.array([[10, 10], [49, 10], [49, 49], [10, 49]], float))
    assert 1500 <= m.sum() <= 1700      # 40x40 = 1600 근처


def test_ring_mask_excludes_center():
    m = ring_mask((101, 101), (50, 50), 10, 30)
    assert m[50, 50] == 0 and m[50, 50 + 20] == 1 and m[50, 50 + 40] == 0


def test_pixels_in_mask_picks_only_masked():
    img = np.zeros((50, 50, 3), np.uint8)
    img[10:20, 10:20] = SKIN_BGR
    mask = np.zeros((50, 50), np.uint8)
    mask[10:20, 10:20] = 1
    px = pixels_in_mask(img, mask)
    assert px.shape == (100, 3) and (px == SKIN_BGR).all()


# ───────────── [B] 실제 사진 (있을 때만) ─────────────
SAMPLES = {
    "photo1": (ROOT / "data/samples/photo1.jpg", (645, 1817, 725, 1897)),
    "photo3": (ROOT / "data/samples/photo3.jpg", (745, 3130, 825, 3210)),
}
need_photos = pytest.mark.skipif(not all(p.is_file() for p, _ in SAMPLES.values()),
                                 reason="data/samples 에 photo1.jpg, photo3.jpg 가 없음")


@pytest.fixture(scope="module")
def real_results():
    pytest.importorskip("mediapipe")
    from backend.image_io import imread
    from backend.vision.face_color import extract_face_colors
    from backend.vision.lighting import white_balance
    out = {}
    for name, (path, white) in SAMPLES.items():
        img = imread(path)
        before = extract_face_colors(img)
        after = extract_face_colors(white_balance(img, white).image, landmarks=before.landmarks)
        out[name] = (before, after)
    return out


@need_photos
def test_real_face_found_and_regions_used(real_results):
    for name, (_, after) in real_results.items():
        assert after.landmarks.num_faces == 1, name
        assert after.regions["cheek_right"].used and after.regions["cheek_left"].used, name
        assert after.skin is not None and after.eye is not None, name


@need_photos
def test_real_bangs_forehead_excluded(real_results):
    """두 사진 모두 앞머리가 이마를 가림 → 이마는 자동 제외되어야 한다"""
    for name, (_, after) in real_results.items():
        assert not after.regions["forehead"].used, name


@need_photos
def test_real_skin_color_converges_after_white_balance(real_results):
    """핵심: 조명이 다른 두 사진의 피부색(a·b)이 보정 후 가까워져야 한다"""
    (b1, a1), (b3, a3) = real_results["photo1"], real_results["photo3"]
    ab = lambda x, y: float(np.linalg.norm((x - y)[1:]))   # noqa: E731
    assert ab(a1.skin, a3.skin) < ab(b1.skin, b3.skin) * 0.5
