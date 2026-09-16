"""
=====================================================================
 face_color.py — [3단계-②] 얼굴 부위별 평균색(Lab) 자동 추출
=====================================================================
 지금까지:  find_coords.py 로 볼 좌표를 '손으로 클릭' → 사각형 평균색
 이제부터:  MediaPipe 랜드마크(landmarks.py) → 부위 영역 자동 생성 → 평균색

 흐름
   보정된 사진 ─▶ ① 랜드마크 478점 ─▶ ② 점 번호로 부위 다각형/원 만들기
               ─▶ ③ 영역 안 픽셀 모으기 ─▶ ④ 이상한 픽셀 걸러내기 ─▶ ⑤ 평균 Lab

 부위 (1주차 범위)
   - 피부 : 오른쪽 볼 + 왼쪽 볼 + 이마
   - 눈   : 양쪽 홍채 (눈꺼풀·동공·반사광 제외)
   - 머리 : 2주차 예정 → 지금은 None

 ④ 걸러내기가 중요한 이유
   볼 영역 안에도 번들거림(하얀 반사광), 그림자, 잔머리, 점이 섞인다.
   → 밝기(L) 기준 하위 10% · 상위 10% 픽셀을 버리고(= 절사평균) 나머지만 평균낸다.
   이마는 앞머리에 가려지는 경우가 많아서, 볼과 색이 너무 다르면 자동으로 제외한다.

 입력 사진은 반드시 lighting.white_balance() 로 '보정된' 사진이어야 한다.
 (랜드마크는 원본에서 찾아도 되지만, 색은 보정본에서 뽑아야 조명 영향이 줄어든다)

 확인 도구: python tools/check_face_color.py --photo data/samples/photo1.jpg --white 645,1817,725,1897
=====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from backend.color_utils import bgr_pixels_to_lab, delta_e
from backend.vision.landmarks import FaceLandmarks, detect_face_landmarks

# ---------------------------------------------------------------------
# 부위별 랜드마크 번호 (MediaPipe FaceMesh 고정 번호)
#   "오른쪽/왼쪽" = 사진 속 사람 기준. (사진에서는 좌우가 반대로 보임)
#   번호를 바꾸고 싶으면 tools/draw_landmarks.py --numbers 로 확인 후 수정.
# ---------------------------------------------------------------------
REGIONS = {
    #          눈 밑 ─────────────▶ 코 옆 ─▶ 볼 중앙 ─▶ 바깥쪽
    "cheek_right": [117, 118, 101, 36, 205, 187, 123],
    "cheek_left":  [346, 347, 330, 266, 425, 411, 352],
    #          이마 위쪽 가로줄 ────────────────▶ 눈썹 바로 위 가로줄(역순)
    "forehead":    [103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104],
}
# 눈 윤곽(눈꺼풀 안쪽 경계) — 홍채 원 중에서 눈꺼풀에 가려진 부분을 잘라내는 데 사용
EYE_CONTOURS = {
    "eye_right": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "eye_left":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
}
# 홍채: 중심점 1개 + 테두리 4개
IRIS = {"iris_a": (468, [469, 470, 471, 472]), "iris_b": (473, [474, 475, 476, 477])}

# ---------------------------------------------------------------------
# 튜닝용 숫자 모음
# ---------------------------------------------------------------------
PARAMS = {
    "trim_low": 10,             # 밝기 하위 몇 % 버릴지 (그림자·잔머리·점)
    "trim_high": 90,            # 밝기 상위 몇 % 이상 버릴지 (번들거림)
    "saturated": 250,           # 이 값 이상 채널이 있는 픽셀은 버림 (하얗게 날아간 반사광)
    "min_pixels": 30,           # 거르고 난 픽셀이 이보다 적으면 그 부위는 '사용 안 함'
    "forehead_max_de": 12.0,    # 이마 색이 두 볼 평균과 ΔE 이만큼 넘게 다르면 → 앞머리로 보고 제외
    "cheek_pair_warn_de": 8.0,  # 양 볼 색 차이가 이보다 크면 → 한쪽만 그늘졌다고 경고
    "iris_inner": 0.35,         # 홍채 반지름 중 안쪽 35% 는 동공(검정)으로 보고 제외
    "iris_outer": 0.90,         # 바깥 10% 는 흰자와 섞이는 경계라 제외
    "min_face_width_ratio": 0.15,  # 얼굴 폭이 사진 폭의 15% 미만이면 '얼굴이 작다' 경고
}


@dataclass
class RegionColor:
    """부위 하나의 추출 결과"""
    lab: np.ndarray | None      # 평균 Lab (사용 불가면 None)
    n_total: int                # 영역 안 전체 픽셀 수
    n_used: int                 # 걸러내고 남은 픽셀 수
    used: bool                  # 최종 평균에 포함했는가
    note: str = ""              # 제외 이유 등

    def to_dict(self) -> dict:
        return {"lab": None if self.lab is None else [round(float(v), 1) for v in self.lab],
                "n_total": self.n_total, "n_used": self.n_used, "used": self.used, "note": self.note}


@dataclass
class FaceColorResult:
    """
    extract_face_colors() 결과 — 알고리즘 파트로 넘기는 데이터.
    to_dict() 형식이 팀 합의 '데이터 형식' (docs/data_format.md) 과 같아야 한다.
    """
    skin: np.ndarray | None                     # 피부 대표 Lab (볼 + [이마])
    eye: np.ndarray | None                      # 눈동자 대표 Lab
    hair: np.ndarray | None                     # 머리카락 Lab (2주차 예정 → 현재 None)
    regions: dict[str, RegionColor]             # 부위별 상세
    landmarks: FaceLandmarks                    # 디버그·시각화용
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 1) for x in v]  # noqa: E731
        return {
            "skin": r(self.skin), "eye": r(self.eye), "hair": r(self.hair),
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "face": {"box": list(self.landmarks.face_box), "width_ratio": round(self.landmarks.face_width_ratio, 3),
                     "num_faces": self.landmarks.num_faces},
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------
# 영역 → 픽셀 모으기
# ---------------------------------------------------------------------
def polygon_mask(shape, pts: np.ndarray) -> np.ndarray:
    """점들을 이은 다각형 내부 = 1 인 마스크 (사진과 같은 크기, uint8)"""
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask


def ring_mask(shape, center, r_in: float, r_out: float) -> np.ndarray:
    """도넛 모양 마스크 (홍채: 동공 빼고 테두리 빼고)"""
    mask = np.zeros(shape[:2], np.uint8)
    c = tuple(int(round(v)) for v in center)
    cv2.circle(mask, c, max(1, int(round(r_out))), 1, -1)
    cv2.circle(mask, c, int(round(r_in)), 0, -1)
    return mask


def pixels_in_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크가 1인 곳의 픽셀만 (N, 3) 으로 꺼낸다. 속도를 위해 마스크 범위만 잘라서 처리."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return np.empty((0, 3), np.uint8)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    sub = img_bgr[y1:y2, x1:x2]
    return sub[mask[y1:y2, x1:x2].astype(bool)]


def robust_mean_lab(pixels_bgr: np.ndarray, trim_low=None, trim_high=None, saturated=None):
    """
    이상한 픽셀을 걸러내고 평균 Lab 을 구한다.  → (평균 Lab 또는 None, 사용한 픽셀 수)
      1) 채널 하나라도 saturated(250) 이상 = 반사광 → 버림
      2) 남은 픽셀 중 밝기(L) 하위 trim_low% / 상위 trim_high% 이상 → 버림
    """
    p = PARAMS
    trim_low = p["trim_low"] if trim_low is None else trim_low
    trim_high = p["trim_high"] if trim_high is None else trim_high
    saturated = p["saturated"] if saturated is None else saturated

    if len(pixels_bgr) == 0:
        return None, 0
    px = pixels_bgr[(pixels_bgr < saturated).all(axis=1)]
    if len(px) == 0:
        return None, 0
    lab = bgr_pixels_to_lab(px)
    lo, hi = np.percentile(lab[:, 0], [trim_low, trim_high])
    keep = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    if len(keep) == 0:
        return None, 0
    return keep.mean(axis=0), len(keep)


# ---------------------------------------------------------------------
# 부위별 추출
# ---------------------------------------------------------------------
def _region(img, pts) -> RegionColor:
    px = pixels_in_mask(img, polygon_mask(img.shape, pts))
    lab, n = robust_mean_lab(px)
    ok = lab is not None and n >= PARAMS["min_pixels"]
    return RegionColor(lab=lab if ok else None, n_total=len(px), n_used=n, used=ok,
                       note="" if ok else "유효 픽셀 부족")


def _iris(img, lm: FaceLandmarks, name: str) -> RegionColor:
    center_idx, rim_idx = IRIS[name]
    c = lm.points[center_idx]
    r = float(np.linalg.norm(lm.points[rim_idx] - c, axis=1).mean())    # 홍채 반지름 = 중심~테두리 평균 거리
    mask = ring_mask(img.shape, c, r * PARAMS["iris_inner"], r * PARAMS["iris_outer"])

    # 이 홍채가 들어 있는 눈의 윤곽(눈꺼풀)으로 한 번 더 자르기 → 눈꺼풀·속눈썹 픽셀 제외
    for contour in EYE_CONTOURS.values():
        poly = lm.points[contour]
        if cv2.pointPolygonTest(poly.astype(np.float32), (float(c[0]), float(c[1])), False) >= 0:
            mask &= polygon_mask(img.shape, poly)
            break

    px = pixels_in_mask(img, mask)
    # 홍채는 영역이 작아서 최소 픽셀 수를 낮춰 적용 (사진 해상도가 낮으면 수십 픽셀 수준)
    lab, n = robust_mean_lab(px)
    ok = lab is not None and n >= max(5, PARAMS["min_pixels"] // 3)
    note = "" if ok else f"유효 픽셀 부족 (홍채 반지름 {r:.1f}px)"
    return RegionColor(lab=lab if ok else None, n_total=len(px), n_used=n, used=ok, note=note)


def extract_face_colors(img_bgr: np.ndarray, landmarks: FaceLandmarks | None = None) -> FaceColorResult:
    """
    ★ 보통 이 함수 하나만 쓰면 된다 ★
    보정된 사진에서 피부·눈 대표 Lab 을 자동으로 뽑는다.

    img_bgr   : lighting.white_balance(...).image  (보정된 사진)
    landmarks : 이미 찾은 랜드마크가 있으면 넘겨서 재사용 (없으면 여기서 찾음)

    사용 예)
        from backend.image_io import imread
        from backend.vision.lighting import white_balance
        from backend.vision.face_color import extract_face_colors

        img = imread("data/samples/photo1.jpg")
        wb = white_balance(img, (645, 1817, 725, 1897))
        fc = extract_face_colors(wb.image)
        print(fc.skin, fc.eye, fc.warnings)
    """
    lm = landmarks or detect_face_landmarks(img_bgr)
    P = lm.points
    warnings: list[str] = []

    regions: dict[str, RegionColor] = {name: _region(img_bgr, P[idx]) for name, idx in REGIONS.items()}
    for name in IRIS:
        regions[name] = _iris(img_bgr, lm, name)

    # ── 피부 대표색: 두 볼을 기준으로, 이마는 볼과 비슷할 때만 포함 ──
    cheeks = [regions[k] for k in ("cheek_right", "cheek_left") if regions[k].used]
    skin = None
    if cheeks:
        cheek_mean = np.mean([c.lab for c in cheeks], axis=0)
        if len(cheeks) == 2 and delta_e(cheeks[0].lab, cheeks[1].lab) > PARAMS["cheek_pair_warn_de"]:
            warnings.append(f"양 볼 색 차이가 큽니다 (ΔE {delta_e(cheeks[0].lab, cheeks[1].lab):.1f}). "
                            "한쪽에 그림자가 졌을 수 있어요. 정면 조명에서 촬영해주세요.")
        fh = regions["forehead"]
        if fh.used:
            d = delta_e(fh.lab, cheek_mean)
            if d > PARAMS["forehead_max_de"]:
                fh.used = False
                fh.note = f"볼과 색 차이 ΔE {d:.1f} → 앞머리·그림자로 보고 제외"
        parts = cheeks + ([fh] if fh.used else [])
        # 픽셀 수로 가중평균 (넓은 부위가 더 큰 비중)
        skin = np.average([p.lab for p in parts], axis=0, weights=[p.n_used for p in parts])
    else:
        warnings.append("볼 영역에서 피부색을 추출하지 못했습니다.")

    # ── 눈 대표색: 쓸 수 있는 홍채들의 평균 ──
    irises = [regions[k] for k in IRIS if regions[k].used]
    eye = np.average([i.lab for i in irises], axis=0, weights=[i.n_used for i in irises]) if irises else None
    if eye is None:
        warnings.append("눈동자 색을 추출하지 못했습니다. 눈을 뜨고 정면을 봐주세요.")

    if lm.num_faces > 1:
        warnings.append(f"얼굴이 {lm.num_faces}명 보입니다. 가장 큰 얼굴로 진단했어요.")
    if lm.face_width_ratio < PARAMS["min_face_width_ratio"]:
        warnings.append(f"얼굴이 작게 찍혔습니다 (사진 폭의 {lm.face_width_ratio:.0%}). 조금 더 가까이서 촬영해주세요.")

    return FaceColorResult(skin=skin, eye=eye, hair=None, regions=regions, landmarks=lm, warnings=warnings)


# ---------------------------------------------------------------------
# 시각화 (확인 도구·웹 디버그 화면용)
# ---------------------------------------------------------------------
def draw_regions(img_bgr: np.ndarray, result: FaceColorResult) -> np.ndarray:
    """추출에 쓴 영역을 사진 위에 그린 복사본. 초록=사용 / 빨강=제외"""
    out = img_bgr.copy()
    P = result.landmarks.points
    t = max(2, int((result.landmarks.face_box[2] - result.landmarks.face_box[0]) / 250))
    for name, idx in REGIONS.items():
        color = (0, 200, 0) if result.regions[name].used else (0, 0, 255)
        cv2.polylines(out, [np.round(P[idx]).astype(np.int32)], True, color, t)
    for name, (ci, rim) in IRIS.items():
        c = P[ci]
        r = float(np.linalg.norm(P[rim] - c, axis=1).mean())
        color = (0, 200, 0) if result.regions[name].used else (0, 0, 255)
        cc = tuple(int(round(v)) for v in c)
        cv2.circle(out, cc, int(round(r * PARAMS["iris_outer"])), color, max(1, t // 2))
        cv2.circle(out, cc, int(round(r * PARAMS["iris_inner"])), color, max(1, t // 2))
    for contour in EYE_CONTOURS.values():
        cv2.polylines(out, [np.round(P[contour]).astype(np.int32)], True, (255, 200, 0), max(1, t // 2))
    return out
