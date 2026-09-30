"""
=====================================================================
 face_color.py — [3단계-②] 얼굴 부위별 평균색(Lab) 자동 추출
=====================================================================
 지금까지:  find_coords.py 로 볼 좌표를 '손으로 클릭' → 사각형 평균색
 이제부터:  MediaPipe 랜드마크(landmarks.py) → 부위 영역 자동 생성 → 평균색

 흐름
   보정된 사진 ─▶ ① 랜드마크 478점 ─▶ ② 점 번호로 부위 다각형/원 만들기
               ─▶ ③ 영역 안 픽셀 모으기 ─▶ ④ 이상한 픽셀 걸러내기 ─▶ ⑤ 평균 Lab

 부위
   - 피부 : 오른쪽 볼 + 왼쪽 볼 + 이마
   - 눈   : 양쪽 홍채 (눈꺼풀·동공·반사광 제외)
   - 머리 : 머리카락 — 이마 위 띠 영역 + (앞머리로 가려진 이마) 중 '피부보다 어두운' 픽셀

 사진 품질 지표도 같이 계산한다 (5단계 신뢰도 재료)
   밝기(피부 L) · 선명도(라플라시안 분산) · 얼굴 크기 · 좌우 비대칭(옆얼굴 판단)

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
# 머리카락: 이마 위쪽 가로줄 → 이를 위로 평행이동해 '머리 띠' 영역을 만든다
#            (MediaPipe 에는 머리카락 점이 없어서 직접 만들어야 함)
HAIRLINE = [103, 67, 109, 10, 338, 297, 332]

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
    "use_forehead_in_skin": False,
    "_use_forehead_근거": "이마를 피부 대표색에 넣으면 웜/쿨 구분이 나빠진다. 실측: "
                          "라벨 확실한 웹캠 4장 간격 +4.3°→+6.8°, 팀원 촬영본(경계형 vs 쿨) -0.1°→+4.2°. "
                          "앞머리 그림자·이마 번들거림이 섞이기 때문. 볼이 둘 다 실패할 때만 이마를 쓴다",
    "forehead_max_de": 12.0,    # 이마 색이 두 볼 평균과 ΔE 이만큼 넘게 다르면 → 앞머리로 보고 제외
    "cheek_pair_warn_de": 8.0,  # 양 볼 색 차이가 이보다 크면 → 한쪽만 그늘졌다고 경고
    "iris_inner": 0.35,         # 홍채 반지름 중 안쪽 35% 는 동공(검정)으로 보고 제외
    "iris_outer": 0.90,         # 바깥 10% 는 흰자와 섞이는 경계라 제외
    "iris_darker_than_skin": 10.0,  # 홍채는 피부보다 최소 이만큼 어둡다. 아니면 눈꺼풀을 잡은 것
    "min_face_width_ratio": 0.15,  # 얼굴 폭이 사진 폭의 15% 미만이면 '얼굴이 작다' 경고
    # ── 머리카락 ──
    "hair_band_ratio": 0.18,    # 이마 위로 얼굴 높이의 18% 만큼을 '머리 띠'로 본다
    "hair_darker_than_skin": 15.0,  # 피부보다 L 이 이만큼 어두운 픽셀만 머리카락으로 인정
    "hair_min_dark_ratio": 0.20,    # 띠 영역에서 어두운 픽셀이 25% 미만이면 → 배경만 잡힌 것으로 보고 제외
    # ── 사진 품질 (5단계 신뢰도 재료) ──
    "min_sharpness": 10.0,      # 얼굴을 가로 400px 로 맞춘 뒤 라플라시안 분산. 이보다 낮으면 흐림
    "min_brightness": 35.0,     # 피부 L 하한 (너무 어두운 사진)
    "max_brightness": 88.0,     # 피부 L 상한 (하얗게 날아간 사진)
    "max_asymmetry": 0.15,      # 코~좌우 얼굴 끝 거리 차이 / 얼굴 폭. 크면 옆얼굴
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
class PhotoQuality:
    """사진 품질 지표 — 5단계 신뢰도 계산에 그대로 넘긴다"""
    brightness: float      # 피부 밝기 L (0~100)
    sharpness: float       # 선명도. 얼굴을 가로 400px 로 맞춘 뒤 라플라시안 분산 (클수록 또렷)
    face_width_ratio: float
    asymmetry: float       # 좌우 비대칭 (0 에 가까울수록 정면)

    def to_dict(self) -> dict:
        return {"brightness": round(self.brightness, 1), "sharpness": round(self.sharpness, 1),
                "face_width_ratio": round(self.face_width_ratio, 3), "asymmetry": round(self.asymmetry, 3)}


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
    quality: PhotoQuality                       # 사진 품질 지표
    landmarks: FaceLandmarks                    # 디버그·시각화용
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        r = lambda v: None if v is None else [round(float(x), 1) for x in v]  # noqa: E731
        return {
            "skin": r(self.skin), "eye": r(self.eye), "hair": r(self.hair),
            "regions": {k: v.to_dict() for k, v in self.regions.items()},
            "quality": self.quality.to_dict(),
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


def _hair_band_polygon(lm: FaceLandmarks) -> np.ndarray:
    """이마 위쪽 가로줄을 얼굴 높이의 일정 비율만큼 위로 올려 '머리 띠' 다각형을 만든다."""
    x1, y1, x2, y2 = lm.face_box
    up = (y2 - y1) * PARAMS["hair_band_ratio"]
    line = lm.points[HAIRLINE]
    return np.vstack([line, (line - np.array([0, up], np.float32))[::-1]])


def _hair(img, lm: FaceLandmarks, skin_lab, include_forehead: bool) -> RegionColor:
    """
    머리카락 색. 머리 띠 + (앞머리로 제외된 이마) 안에서 '피부보다 충분히 어두운' 픽셀만 사용.
      - 어두운 픽셀만 쓰는 이유 : 띠 안에 배경(벽·하늘)이 섞이기 때문
      - 피부색을 모르면(볼 추출 실패) 머리색도 구하지 않는다
    """
    if skin_lab is None:
        return RegionColor(lab=None, n_total=0, n_used=0, used=False, note="피부색을 몰라 머리색 판단 불가")

    mask = polygon_mask(img.shape, _hair_band_polygon(lm))
    # 이마가 '앞머리에 가려져 제외된' 경우에만 이마 영역도 머리카락 후보에 넣는다.
    # (이마가 피부로 잘 잡힌 사람은 넣으면 안 됨 — 밝은 피부가 섞여 머리 탐색을 방해한다)
    if include_forehead:
        mask |= polygon_mask(img.shape, lm.points[REGIONS["forehead"]])
    px = pixels_in_mask(img, mask)
    if len(px) == 0:
        return RegionColor(lab=None, n_total=0, n_used=0, used=False, note="머리 영역이 사진 밖")

    lab = bgr_pixels_to_lab(px)
    dark = px[lab[:, 0] < skin_lab[0] - PARAMS["hair_darker_than_skin"]]
    ratio = len(dark) / len(px)
    if ratio < PARAMS["hair_min_dark_ratio"]:
        return RegionColor(lab=None, n_total=len(px), n_used=len(dark), used=False,
                           note=f"어두운 픽셀 {ratio:.0%} → 머리카락을 찾지 못함 (배경만 잡힘)")
    hair_lab, n = robust_mean_lab(dark)
    ok = hair_lab is not None and n >= PARAMS["min_pixels"]
    return RegionColor(lab=hair_lab if ok else None, n_total=len(px), n_used=n, used=ok,
                       note="" if ok else "유효 픽셀 부족")


def _quality(img, lm: FaceLandmarks, skin_lab) -> PhotoQuality:
    """사진 품질 지표 4가지"""
    x1, y1, x2, y2 = lm.face_box
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return PhotoQuality(0.0, 0.0, lm.face_width_ratio, 0.0)
    # 얼굴 크기가 사진마다 달라도 같은 기준이 되도록 가로 400px 로 맞춘 뒤 선명도 측정
    h = max(1, int(400 * crop.shape[0] / crop.shape[1]))
    gray = cv2.cvtColor(cv2.resize(crop, (400, h)), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    nose, left, right = lm.points[1], lm.points[234], lm.points[454]   # 코끝 / 얼굴 왼쪽 끝 / 오른쪽 끝
    width = max(1.0, float(x2 - x1))
    asym = abs(float(np.linalg.norm(nose - left) - np.linalg.norm(nose - right))) / width

    brightness = float(skin_lab[0]) if skin_lab is not None else 0.0
    return PhotoQuality(brightness=brightness, sharpness=sharpness,
                        face_width_ratio=lm.face_width_ratio, asymmetry=asym)


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
        # 피부 대표색은 볼만 쓴다 (이마는 앞머리·번들거림이 섞여 웜/쿨 구분을 흐린다)
        parts = cheeks + ([fh] if (fh.used and PARAMS["use_forehead_in_skin"]) else [])
        # 픽셀 수로 가중평균 (넓은 부위가 더 큰 비중)
        skin = np.average([p.lab for p in parts], axis=0, weights=[p.n_used for p in parts])
    elif regions["forehead"].used:
        # 볼을 둘 다 못 쓸 때만 이마로 대신한다
        skin = regions["forehead"].lab
        warnings.append("볼에서 피부색을 못 읽어 이마로 대신했습니다. 정확도가 떨어집니다.")
    else:
        warnings.append("볼 영역에서 피부색을 추출하지 못했습니다.")

    # ── 눈 대표색: 쓸 수 있는 홍채들의 평균 ──
    # 눈을 감았거나 랜드마크가 밀리면 눈꺼풀(살색)을 홍채로 잡는다.
    # 홍채는 피부보다 확실히 어두우므로, 그렇지 않으면 실패로 처리한다.
    if skin is not None:
        for k in IRIS:
            r = regions[k]
            if r.used and r.lab[0] > skin[0] - PARAMS["iris_darker_than_skin"]:
                r.used = False
                r.note = f"피부(L {skin[0]:.0f})보다 어둡지 않음 → 눈꺼풀을 잡은 것으로 보고 제외"
    irises = [regions[k] for k in IRIS if regions[k].used]
    eye = np.average([i.lab for i in irises], axis=0, weights=[i.n_used for i in irises]) if irises else None
    if eye is None:
        warnings.append("눈동자 색을 추출하지 못했습니다. 눈을 뜨고 정면을 봐주세요.")

    # ── 머리카락 ──
    regions["hair"] = _hair(img_bgr, lm, skin, include_forehead=not regions["forehead"].used)
    hair = regions["hair"].lab if regions["hair"].used else None
    if hair is None and skin is not None:
        warnings.append("머리카락 색을 추출하지 못했습니다. " + regions["hair"].note)

    # ── 사진 품질 + 품질 경고 ──
    quality = _quality(img_bgr, lm, skin)
    if quality.sharpness < PARAMS["min_sharpness"]:
        warnings.append(f"사진이 흐립니다 (선명도 {quality.sharpness:.0f}). 흔들리지 않게 다시 촬영해주세요.")
    if skin is not None and quality.brightness < PARAMS["min_brightness"]:
        warnings.append(f"사진이 어둡습니다 (피부 밝기 {quality.brightness:.0f}). 밝은 곳에서 촬영해주세요.")
    if skin is not None and quality.brightness > PARAMS["max_brightness"]:
        warnings.append(f"사진이 너무 밝습니다 (피부 밝기 {quality.brightness:.0f}). 빛이 직접 닿지 않게 해주세요.")
    if quality.asymmetry > PARAMS["max_asymmetry"]:
        warnings.append(f"얼굴이 옆으로 돌아가 있습니다 (비대칭 {quality.asymmetry:.2f}). 정면을 봐주세요.")

    if lm.num_faces > 1:
        warnings.append(f"얼굴이 {lm.num_faces}명 보입니다. 가장 큰 얼굴로 진단했어요.")
    if lm.face_width_ratio < PARAMS["min_face_width_ratio"]:
        warnings.append(f"얼굴이 작게 찍혔습니다 (사진 폭의 {lm.face_width_ratio:.0%}). 조금 더 가까이서 촬영해주세요.")

    return FaceColorResult(skin=skin, eye=eye, hair=hair, regions=regions, quality=quality,
                           landmarks=lm, warnings=warnings)


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
    hair_color = (0, 200, 0) if result.regions.get("hair") and result.regions["hair"].used else (0, 0, 255)
    cv2.polylines(out, [np.round(_hair_band_polygon(result.landmarks)).astype(np.int32)], True, hair_color, t)
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
