"""
=====================================================================
 report.py — 진단 결과 → 결과 요약 리포트 데이터
=====================================================================
 촬영·진단이 끝나면 QR 로 폰에서 열어볼 '내 퍼스널컬러 리포트' 를 만든다.

   result.json  ─▶  build_report()  ─▶  리포트 dict  ─▶  report.html

 하는 일
   ① 시즌별 설명 자료(SEASONS)를 붙인다
   ② 내 피부 Lab 과 각 색을 '웜/명도/채도' 3축으로 비교해 Best·Worst 를 고른다
      └ 축 정의는 season_reference.json 의 seasons[*].{warm,light,chroma} 와 같다
   ③ 왜 그런지를 측정값 기반 문장으로 만든다 (칙칙해 보임 / 또렷해 보임 ...)
   ④ 얼굴을 Best 색·Worst 색 배경에 올린 비교 사진 2장을 만든다

 ※ 비교 사진에서 '얼굴 픽셀 자체는 완전히 같다'.
   옆에 놓인 색 때문에 달라 보이는 것(대비 효과)이고, 실제 드레이핑도 같은 원리다.
   리포트에 이 설명을 반드시 같이 보여준다 — 안 그러면 "얼굴 똑같은데?" 가 된다.
=====================================================================
"""
from __future__ import annotations

import io
import math
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── 시즌별 설명 자료 ───────────────────────────────────────────
#   화면에 그대로 뿌리는 텍스트. 색 이름 옆 hex 는 점수 계산과 색칩에 쓴다.
SEASONS = {
    "spring_warm": {
        "label": "봄 웜",
        "keywords": "따뜻함 · 밝음 · 생기",
        "headline": "밝고 따뜻하며 맑고 생기 있는 색",
        "image": "생기 있고 화사하며 부드러운 느낌",
        "likeness": "봄꽃, 햇살, 복숭아처럼 밝고 화사한 느낌",
        "features": {
            "피부": "노란빛이나 복숭아빛이 느껴지는 경우가 많음",
            "눈동자": "밝은 갈색, 따뜻한 갈색 계열",
            "머리카락": "밝은 갈색부터 자연스러운 갈색까지",
        },
        "palette": [
            ("코랄 레드", "#FF6B4A"), ("살구 · 피치", "#FFB088"), ("크림 옐로", "#FFE08A"),
            ("연두 · 라이트그린", "#A8D672"), ("아쿠아 블루", "#7FD4D0"),
            ("코랄 핑크", "#FF9E8E"), ("밝은 카멜", "#D2A679"),
        ],
        "makeup": {"립": "코랄, 피치, 살구색", "블러셔": "복숭아, 살구색",
                   "아이섀도": "골드 베이지, 코랄 브라운", "헤어": "밀크 브라운, 골드 브라운"},
        "avoid": [("진한 검정", "#111111"), ("탁한 회색", "#8A8A8A"), ("차가운 진보라", "#4B2E83")],
        "subtypes": [("봄 라이트", "밝고 부드러운 파스텔 계열"), ("봄 브라이트", "밝으면서 채도가 높은 선명한 색")],
    },
    "summer_cool": {
        "label": "여름 쿨",
        "keywords": "차가움 · 밝음 · 부드러움",
        "headline": "차갑고 밝으며 부드럽고 은은한 색",
        "image": "청순하고 차분하며 깨끗한 느낌",
        "likeness": "라벤더, 하늘색, 장미색처럼 시원하면서도 부드러운 느낌",
        "features": {
            "피부": "핑크빛이나 중성적인 느낌",
            "눈동자": "부드러운 갈색, 회갈색 계열",
            "머리카락": "애쉬 브라운, 자연스러운 어두운 갈색",
        },
        "palette": [
            ("로즈 핑크", "#E89BAE"), ("베이비 핑크", "#F4C2CE"), ("더스티 핑크", "#D4A5AF"),
            ("라벤더", "#C9B6E4"), ("스카이 블루", "#A8DCE7"), ("파우더 블루", "#8FB8DE"),
            ("민트", "#9FD9C7"), ("쿨 그레이", "#C7CED4"),
        ],
        "makeup": {"립": "로즈 핑크, 모브 핑크, 말린 장미", "블러셔": "연한 핑크, 라벤더 핑크",
                   "아이섀도": "회갈색, 로즈 브라운", "헤어": "애쉬 브라운, 쿨 브라운"},
        "avoid": [("진한 주황", "#E2621B"), ("노란기 강한 카멜", "#B07A2A"), ("형광색", "#CCFF00")],
        "subtypes": [("여름 라이트", "밝고 맑은 파스텔 색"), ("여름 뮤트", "회색이 섞인 차분하고 부드러운 색")],
    },
    "autumn_warm": {
        "label": "가을 웜",
        "keywords": "따뜻함 · 깊음 · 차분함",
        "headline": "따뜻하고 깊으며 차분하고 자연스러운 색",
        "image": "분위기 있고 성숙하며 고급스러운 느낌",
        "likeness": "낙엽, 나무, 흙, 단풍처럼 깊이 있고 자연스러운 느낌",
        "features": {
            "피부": "노란빛, 베이지빛, 황금빛이 조화로움",
            "눈동자": "짙은 갈색, 골드 브라운",
            "머리카락": "다크 브라운, 초콜릿 브라운",
        },
        "palette": [
            ("브릭 레드", "#A33B2A"), ("테라코타", "#C1573B"), ("머스터드", "#D9A441"),
            ("올리브", "#6B7A3A"), ("카키", "#7A7549"), ("딥 틸", "#2F6B6B"),
            ("카멜", "#B88354"), ("초콜릿", "#5C4033"), ("웜 베이지", "#D5BA96"),
        ],
        "makeup": {"립": "브릭 레드, 칠리, 말린 오렌지", "블러셔": "누드 피치, 살구 브라운",
                   "아이섀도": "골드 브라운, 카키 브라운", "헤어": "초콜릿 브라운, 다크 카라멜"},
        "avoid": [("형광 핑크", "#FF3FA4"), ("차가운 라벤더", "#C9B6E4"), ("차가운 파스텔", "#BFE3F5")],
        "subtypes": [("가을 뮤트", "부드럽고 탁한 어스 컬러"), ("가을 딥", "어둡고 깊은 브라운·버건디·딥그린")],
    },
    "winter_cool": {
        "label": "겨울 쿨",
        "keywords": "차가움 · 선명함 · 강렬함",
        "headline": "차갑고 선명하며 대비가 강한 색",
        "image": "도시적이고 세련되며 선명한 느낌",
        "likeness": "눈, 얼음, 보석처럼 선명하고 강렬한 느낌",
        "features": {
            "피부": "핑크빛이나 차가운 느낌",
            "눈동자": "짙은 갈색, 검정에 가까운 색",
            "머리카락": "자연 흑발, 짙은 흑갈색",
        },
        "palette": [
            ("체리 레드", "#BE1E3C"), ("핫핑크", "#E8176B"), ("푸시아", "#D81B8C"),
            ("로열 블루", "#2B5299"), ("코발트 블루", "#1B4FA0"), ("에메랄드", "#00897B"),
            ("바이올렛", "#6A3FA0"), ("순백", "#F7F9FB"), ("블랙", "#1C1C1E"),
        ],
        "makeup": {"립": "체리 레드, 버건디, 푸시아 핑크", "블러셔": "쿨 핑크, 베리 핑크",
                   "아이섀도": "실버, 차콜, 쿨 브라운", "헤어": "블랙, 블루 블랙, 다크 애쉬"},
        "avoid": [("노란기 강한 베이지", "#D9C08A"), ("탁한 오렌지", "#C9752F"), ("황금빛 브라운", "#A8762F")],
        "subtypes": [("겨울 브라이트", "매우 선명하고 맑은 색"), ("겨울 딥", "어둡고 깊은 색, 강한 대비")],
    },
}

# season_reference.json 의 seasons[*] 와 같은 3축 부호
AXES = {
    "spring_warm": {"warm": 1, "light": 1, "chroma": 1},
    "autumn_warm": {"warm": 1, "light": -1, "chroma": -1},
    "summer_cool": {"warm": -1, "light": 1, "chroma": -1},
    "winter_cool": {"warm": -1, "light": -1, "chroma": 1},
}
OPPOSITE = {"spring_warm": "winter_cool", "winter_cool": "spring_warm",
            "autumn_warm": "summer_cool", "summer_cool": "autumn_warm"}


# ── 색 계산 ────────────────────────────────────────────────────
def hex_to_lab(hex_color: str) -> tuple[float, float, float]:
    """#RRGGBB → Lab (L 0~100). color_utils.bgr_to_lab 과 같은 방식(0~1 실수 변환)"""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    px = np.array([[[b, g, r]]], np.float32) / 255.0
    L, A, B = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0]
    return float(L), float(A), float(B)


def color_axes(lab) -> dict:
    """
    옷 색을 피부와 같은 3축으로 환산한다 (대략 -2 ~ +2).

    warm   : 색상각이 주황~노랑(60°) 쪽이면 +, 파랑(240°) 쪽이면 -
    light  : 밝을수록 +
    chroma : 선명할수록 +
    """
    L, a, b = lab
    h = math.degrees(math.atan2(b, a)) % 360
    C = math.hypot(a, b)
    clamp = lambda v: max(-2.0, min(2.0, v))
    return {
        "warm": clamp(2.0 * math.cos(math.radians(h - 60.0))),
        "light": clamp((L - 55.0) / 22.0),
        "chroma": clamp((C - 38.0) / 22.0),
        "_h": round(h, 1), "_L": round(L, 1), "_C": round(C, 1),
    }


# 축별 가중치 — 웜/쿨이 퍼스널컬러의 1차 기준이라 가장 무겁게 둔다
WEIGHTS = {"warm": 2.0, "light": 1.0, "chroma": 1.0}

# 시즌별 '목표 지점'. 부호만 보면 "가을은 어두울수록 좋다" 가 되어 새까만 색이 1등이 된다.
# 목표에서 멀어질수록 깎는 방식이라야 카멜·테라코타 같은 중간 톤이 제대로 올라온다.
TARGETS = {
    "spring_warm": {"warm": 1.2, "light": 0.9, "chroma": 0.8},
    "autumn_warm": {"warm": 1.2, "light": -0.6, "chroma": -0.4},
    "summer_cool": {"warm": -1.0, "light": 0.7, "chroma": -0.7},
    "winter_cool": {"warm": -1.0, "light": -0.5, "chroma": 1.0},
}
SPREAD = 1.6      # 목표에서 이만큼 벗어나면 적합도 0


def score_color(season_key: str, color_lab, skin_lab=None) -> tuple[float, dict]:
    """옷 색의 3축이 내 시즌의 목표 지점에 얼마나 가까운가 (1 에 가까울수록 잘 맞음)"""
    ax = color_axes(color_lab)
    tgt = TARGETS[season_key]
    parts, total = {}, 0.0
    for k, w in WEIGHTS.items():
        diff = ax[k] - tgt[k]
        fit = max(-1.0, min(1.0, 1.0 - abs(diff) / SPREAD))
        parts[k] = fit
        parts[f"{k}_diff"] = round(diff, 2)        # +면 과함, -면 모자람
        total += w * fit
    # 내 피부와의 실제 밝기 차이 — 설명에 숫자로 넣으면 색마다 문장이 달라진다
    parts["dL"] = round(float(skin_lab[0]) - color_lab[0], 1) if skin_lab else None
    return total / sum(WEIGHTS.values()), {**parts, "raw": ax}


# ── 설명 문장 ──────────────────────────────────────────────────
_GOOD = {
    "warm": "언더톤이 같은 방향이라 얼굴색이 자연스럽게 섞입니다",
    "light": "얼굴과 밝기 차이가 적당해 이목구비가 또렷해 보입니다",
    "chroma": "색의 선명함이 피부와 균형이 맞습니다",
}
#  '+' = 내게 맞는 지점보다 과함 / '-' = 모자람
_BAD = {
    "warm": {
        "+": "노란기가 필요 이상으로 강해 얼굴에 누런기가 돌아 보입니다",
        "-": "푸른기가 돌아 얼굴이 창백하고 칙칙해 보입니다",
    },
    "light": {
        "+": "얼굴보다 너무 밝아 상대적으로 얼굴이 어둡고 그늘져 보입니다",
        "-": "너무 어두워서 얼굴과 옷의 경계가 묻히고 이목구비가 흐려집니다",
    },
    "chroma": {
        "+": "색이 너무 선명해 시선이 옷으로 쏠리고 얼굴이 생기 없어 보입니다",
        "-": "색이 탁해서 얼굴까지 같이 칙칙해 보입니다",
    },
}


def explain(season_key: str, parts: dict) -> list[str]:
    """3축 적합도 → 사람이 읽는 이유 목록 (가장 두드러진 것부터 최대 2개)"""
    out = []
    dL = parts.get("dL")
    for k in sorted(WEIGHTS, key=lambda x: -abs(parts[x] - 1) * WEIGHTS[x]):
        fit, diff = parts[k], parts[f"{k}_diff"]
        if k == "light" and dL is not None:
            # 밝기만큼은 '내 피부보다 몇 만큼' 이라고 숫자로 말할 수 있다
            gap, side = abs(dL), ("어두워" if dL > 0 else "밝아")
            if fit >= 0.55:
                out.append(f"내 피부보다 {gap:.0f} 정도 {side} 얼굴 윤곽이 또렷해 보입니다")
            elif fit <= 0.1:
                out.append(f"내 피부보다 {gap:.0f}나 {side} "
                           + ("얼굴이 상대적으로 어둡고 그늘져 보입니다" if dL < 0
                              else "얼굴과 옷 경계가 묻혀 이목구비가 흐려집니다"))
            continue
        if fit >= 0.55:
            out.append(_GOOD[k])
        elif fit <= 0.1:
            out.append(_BAD[k]["+" if diff > 0 else "-"])
    return out[:2] or ["크게 거슬리지 않고 무난하게 받는 편입니다"]


def _rank(season_key: str, items: list[tuple[str, str]], skin_lab=None):
    rows = []
    for name, hx in items:
        lab = hex_to_lab(hx)
        s, parts = score_color(season_key, lab, skin_lab)
        rows.append({"name": name, "hex": hx, "score": round(s, 3),
                     "parts": parts, "why": explain(season_key, parts)})
    return rows


def _dedupe(rows: list[dict]) -> list[dict]:
    """목록에서 같은 설명이 반복되면 읽는 사람이 '복붙' 으로 느낀다.
    이미 쓴 문장은 피하고, 전부 겹치면 그 색은 이유를 생략한다."""
    used = set()
    for r in rows:
        pick = next((w for w in r["why"] if w not in used), None)
        r["why_short"] = [pick] if pick else []
        if pick:
            used.add(pick)
    return rows


def pick_best_worst(season_key: str, n=3, skin_lab=None):
    """추천 색은 내 시즌 팔레트에서, 피할 색은 반대 시즌 + 공식 '피할 색' 에서 고른다"""
    best = sorted(_rank(season_key, SEASONS[season_key]["palette"], skin_lab),
                  key=lambda r: -r["score"])[:n]

    opp = OPPOSITE[season_key]
    pool = SEASONS[opp]["palette"] + SEASONS[season_key]["avoid"]
    worst = sorted(_rank(season_key, pool, skin_lab), key=lambda r: r["score"])[:n]
    return _dedupe(best), _dedupe(worst)


# ── 비교 사진 (디지털 드레이핑) ────────────────────────────────
def make_drape(img_bgr: np.ndarray, hex_color: str, size: int = 560) -> np.ndarray | None:
    """
    색 배경 한가운데 타원으로 얼굴을 올린 정사각형 이미지.
    실제 드레이핑(목·가슴에 천을 대는 것)을 사진으로 재현한 것.
    """
    from backend.vision.landmarks import detect_face_landmarks, FaceNotFoundError
    try:
        lm = detect_face_landmarks(img_bgr)
        x1, y1, x2, y2 = lm.face_box        # property 라 괄호를 붙이면 안 된다
    except (FaceNotFoundError, ValueError, RuntimeError):
        return None

    fw, fh = x2 - x1, y2 - y1
    if fw < 10 or fh < 10:
        return None

    # 얼굴 주변을 조금 넉넉히 (머리·턱선까지)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(fw, fh) * 0.70
    sx1, sy1 = int(cx - half), int(cy - half * 1.08)
    sx2, sy2 = int(cx + half), int(cy + half * 1.08)
    H, W = img_bgr.shape[:2]
    pad = max(0, -sx1, -sy1, sx2 - W, sy2 - H)
    if pad:
        img_bgr = cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        sx1 += pad; sx2 += pad; sy1 += pad; sy2 += pad
    face = img_bgr[sy1:sy2, sx1:sx2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)

    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    canvas = np.full((size, size, 3), (b, g, r), np.uint8)

    # 타원 마스크 + 가장자리 부드럽게
    mask = np.zeros((size, size), np.uint8)
    cv2.ellipse(mask, (size // 2, int(size * 0.46)),
                (int(size * 0.30), int(size * 0.41)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), size * 0.022)
    m3 = (mask.astype(np.float32) / 255.0)[:, :, None]
    return (face * m3 + canvas * (1 - m3)).astype(np.uint8)


# ── 리포트 조립 ────────────────────────────────────────────────
def build_report(result: dict, capture_dir: Path | None = None,
                 corrected_bgr: np.ndarray | None = None,
                 photo_source: str | None = None) -> dict:
    """진단 결과 dict → report.html 이 그대로 쓰는 dict"""
    top = result.get("top") or "autumn_warm"
    if top not in SEASONS:
        top = "autumn_warm"
    info = SEASONS[top]
    detail = result.get("season_detail") or {}
    skin = (result.get("colors") or {}).get("skin") or [0, 0, 0]

    best, worst = pick_best_worst(top, skin_lab=skin)

    # 비교 사진 저장 (실패해도 리포트는 나온다)
    #  한 번 만든 비교 사진은 다시 만들지 않는다.
    #  얼굴 검출이 무거워서, 리포트를 열 때마다 돌리면 페이지가 느려진다.
    drape = {}
    if capture_dir is not None and best and worst:
        for tag, row in (("best", best[0]), ("worst", worst[0])):
            p = capture_dir / f"drape_{tag}.jpg"
            if p.exists():
                drape[tag] = p.name
                continue
            if corrected_bgr is None:
                continue
            img = make_drape(corrected_bgr, row["hex"])
            if img is not None:
                cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                drape[tag] = p.name

    # 측정값을 사람 말로
    h = detail.get("hue_deg")
    th = detail.get("threshold")
    if h is not None and th is not None:
        gap = h - th
        if abs(gap) < 2.5:
            measure = (f"피부 색상각이 {h}°로, 기준선 {th}° 바로 근처입니다. "
                       "웜과 쿨 사이 경계에 있어 양쪽 색을 모두 쓸 수 있습니다.")
        else:
            side = "위" if gap > 0 else "아래"
            tone = "웜" if gap > 0 else "쿨"
            measure = (f"피부 색상각이 {h}°로 기준선 {th}°보다 {abs(gap):.1f}° {side}입니다. "
                       f"{tone} 쪽이 뚜렷합니다.")
    else:
        measure = ""

    return {
        "season_key": top,
        # 비교 사진에 조명 보정본을 썼는가. 원본이면 색이 틀어져 있으니 화면에 알린다
        "photo_corrected": photo_source == "corrected.jpg",
        "info": info,
        "best": best,
        "worst": worst,
        "drape": drape,
        "measure_text": measure,
        "skin_lab": [round(v, 1) for v in skin],
        "skin_hex": lab_to_hex(skin),
        "tone": result.get("tone") or {},
        "percentages": result.get("percentages") or [],
        "reliability": result.get("reliability"),
        "reliability_msg": result.get("reliability_msg"),
        "message": result.get("message"),
        "warnings": result.get("warnings") or [],
        "detail": detail,
    }


def lab_to_hex(lab) -> str:
    """피부 대표색을 화면에 색칩으로 보여주기 위해"""
    try:
        arr = np.array([[[float(lab[0]), float(lab[1]), float(lab[2])]]], np.float32)
        bgr = cv2.cvtColor(arr, cv2.COLOR_LAB2BGR)[0, 0]
        b, g, r = (int(max(0, min(255, round(v * 255)))) for v in bgr)
        return f"#{r:02X}{g:02X}{b:02X}"
    except Exception:
        return "#D8B59A"


# ── 저장용 카드 이미지 ─────────────────────────────────────────
#  QR 로 들어온 사람이 "다운로드" 를 눌러 갤러리에 담아가는 한 장짜리 요약.
#  긴 페이지를 통째로 캡처하는 것보다 공유하기 좋고, 폰에서 바로 저장된다.
from PIL import Image, ImageDraw, ImageFont

#  한글이 나와야 하므로 시스템 폰트를 찾아 쓴다. (경로, ttc 안에서의 번호)
_FONT_CANDIDATES = [
    ("C:/Windows/Fonts/malgunbd.ttf", 0, True), ("C:/Windows/Fonts/malgun.ttf", 0, False),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 1, True),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 1, False),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0, False),
]
_font_cache: dict = {}


def _font(size: int, bold: bool = False):
    key = (size, bold)
    if key in _font_cache:
        return _font_cache[key]
    for path, idx, is_bold in _FONT_CANDIDATES:
        if is_bold != bold or not Path(path).exists():
            continue
        try:
            f = ImageFont.truetype(path, size, index=idx)
            _font_cache[key] = f
            return f
        except Exception:
            pass
    # 굵은 폰트가 없으면 보통 굵기로라도
    for path, idx, _ in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                f = ImageFont.truetype(path, size, index=idx)
                _font_cache[key] = f
                return f
            except Exception:
                pass
    return ImageFont.load_default()


def _hex_rgb(hx: str):
    hx = hx.lstrip("#")
    return tuple(int(hx[i:i + 2], 16) for i in (0, 2, 4))


def _wrap(draw, text, font, max_w):
    """글자 단위로 줄바꿈 (한글은 띄어쓰기가 적어 단어 단위로는 넘친다)"""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def make_card(r: dict, capture_dir: Path, width: int = 1080, as_img: bool = False):
    """리포트 dict → 저장용 PNG 바이트"""
    W, H = width, 1800
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)

    accent = _hex_rgb(r["percentages"][0]["color"]) if r["percentages"] else (242, 121, 157)
    INK, SOFT = (59, 47, 74), (120, 108, 134)

    # 배경 — 위는 크림, 아래로 갈수록 연보라
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)],
               fill=(int(255 - 9 * t), int(247 - 8 * t), int(250 + 2 * t)))

    M = 72                      # 좌우 여백
    y = 92

    d.text((W // 2, y), "COLORY · 퍼스널컬러 진단 결과", font=_font(30), fill=SOFT, anchor="mt")
    y += 62
    d.text((W // 2, y), r["info"]["label"], font=_font(132, True), fill=accent, anchor="mt")
    y += 168
    d.text((W // 2, y), r["info"]["keywords"], font=_font(38), fill=SOFT, anchor="mt")
    y += 68
    d.text((W // 2, y), r["info"]["headline"], font=_font(34), fill=INK, anchor="mt")
    y += 84

    # 칩
    chips = []
    tone = r.get("tone") or {}
    if tone.get("label"):
        pct = tone.get("warm") if tone.get("verdict") != "cool" else tone.get("cool")
        chips.append(f"{tone['label']}톤 {pct}%")
    if r.get("reliability") is not None:
        chips.append(f"신뢰도 {r['reliability']}")
    cf = _font(30)
    widths = [d.textlength(c, font=cf) + 48 for c in chips]
    x = (W - (sum(widths) + 16 * (len(chips) - 1))) / 2
    for c, cw in zip(chips, widths):
        d.rounded_rectangle([x, y, x + cw, y + 58], radius=29,
                            fill=(255, 255, 255), outline=(232, 222, 245), width=2)
        d.text((x + cw / 2, y + 29), c, font=cf, fill=INK, anchor="mm")
        x += cw + 16
    y += 110

    # Best / Worst 비교 사진
    bw = (W - M * 2 - 28) // 2
    for i, (tag, key, col) in enumerate((("잘 받는 색", "best", (31, 122, 83)),
                                         ("안 받는 색", "worst", (176, 66, 60)))):
        px = M + i * (bw + 28)
        row = (r.get(key) or [{}])[0]
        name = row.get("name", "-")
        d.text((px, y), f"{tag} · {name}", font=_font(32, True), fill=col)
        fn = (r.get("drape") or {}).get(key)
        box_y = y + 48
        if fn and (capture_dir / fn).exists():
            ph = Image.open(capture_dir / fn).convert("RGB").resize((bw, bw))
            mask = Image.new("L", (bw, bw), 0)
            ImageDraw.Draw(mask).rounded_rectangle([0, 0, bw, bw], radius=34, fill=255)
            img.paste(ph, (px, box_y), mask)
        else:
            d.rounded_rectangle([px, box_y, px + bw, box_y + bw], radius=34,
                                fill=_hex_rgb(row.get("hex", "#DDDDDD")))
    y += 48 + bw + 26

    # 각 색의 이유 한 줄씩
    rf = _font(29)
    for key, col in (("best", (31, 122, 83)), ("worst", (176, 66, 60))):
        row = (r.get(key) or [{}])[0]
        why = (row.get("why") or [""])[0]
        d.ellipse([M, y + 9, M + 18, y + 27], fill=_hex_rgb(row.get("hex", "#CCCCCC")))
        for ln in _wrap(d, why, rf, W - M * 2 - 34)[:2]:
            d.text((M + 34, y), ln, font=rf, fill=SOFT)
            y += 40
        y += 12
    y += 10

    # 추천 색칩
    d.text((M, y), "나에게 잘 받는 색", font=_font(34, True), fill=INK)
    y += 58
    pal = r["info"]["palette"][:9]
    sw = (W - M * 2 - 14 * (len(pal) - 1)) / len(pal)
    for i, (_, hx) in enumerate(pal):
        sx = M + i * (sw + 14)
        d.rounded_rectangle([sx, y, sx + sw, y + 84], radius=18, fill=_hex_rgb(hx))
    y += 124

    # 안내 문구
    note = ("얼굴 사진 두 장은 완전히 같습니다. 옆에 놓인 색 때문에 다르게 보이는 것이고, "
            "실제 퍼스널컬러 진단에서 천을 대어보는 것과 같은 원리입니다.")
    nf = _font(27)
    nh = len(_wrap(d, note, nf, W - M * 2 - 56)) * 38 + 44
    d.rounded_rectangle([M, y, W - M, y + nh], radius=24,
                        fill=(255, 247, 250), outline=(255, 217, 228), width=2)
    ty = y + 22
    for ln in _wrap(d, note, nf, W - M * 2 - 56):
        d.text((M + 28, ty), ln, font=nf, fill=SOFT)
        ty += 38

    # 하단
    d.text((W // 2, H - 56), f"{datetime.now():%Y.%m.%d}  ·  COLORY",
           font=_font(26), fill=(165, 155, 178), anchor="mb")

    # 아래 남는 여백을 잘라낸다 (내용 길이에 따라 카드 높이가 달라진다)
    bottom = int(ty + 150)
    if bottom < H:
        img = img.crop((0, 0, W, bottom))
        d = ImageDraw.Draw(img)
        d.text((W // 2, bottom - 52), f"{datetime.now():%Y.%m.%d}  ·  COLORY",
               font=_font(26), fill=(165, 155, 178), anchor="mb")

    return img if as_img else _to_png(img)


# ── 전체 리포트를 이미지 여러 장으로 ───────────────────────────
#  카드 한 장으로는 담기지 않는 내용(추천색·피할색·메이크업·특징)까지
#  같은 디자인으로 그려서 갤러리에 저장해 갈 수 있게 한다.
_INK, _SOFT, _LINE = (59, 47, 74), (120, 108, 134), (232, 222, 245)


def _page_bg(W, H):
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / H
        d.line([(0, y), (W, y)], fill=(int(255 - 9 * t), int(247 - 8 * t), int(250 + 2 * t)))
    return img, d


def _page_head(d, W, title, sub, accent, y=78):
    d.text((W // 2, y), title, font=_font(52, True), fill=accent, anchor="mt")
    y += 76
    if sub:
        d.text((W // 2, y), sub, font=_font(30), fill=_SOFT, anchor="mt")
        y += 48
    return y + 22


def _page_foot(img, d, W, bottom, label, idx, total):
    d.text((W // 2, bottom - 54), f"{label}  ·  {idx}/{total}  ·  COLORY",
           font=_font(26), fill=(165, 155, 178), anchor="mb")
    return img.crop((0, 0, W, bottom))


def _swatch_grid(d, items, x0, y, W, M, cols=3):
    """색칩 + 이름 격자"""
    cw = (W - M * 2 - 16 * (cols - 1)) / cols
    f = _font(26)
    for i, (name, hx) in enumerate(items):
        cx = x0 + (i % cols) * (cw + 16)
        cy = y + (i // cols) * 126
        d.rounded_rectangle([cx, cy, cx + cw, cy + 80], radius=18, fill=_hex_rgb(hx))
        for j, ln in enumerate(_wrap(d, name, f, cw)[:2]):
            d.text((cx + cw / 2, cy + 90 + j * 32), ln, font=f, fill=_SOFT, anchor="mt")
    rows = (len(items) + cols - 1) // cols
    return y + rows * 126 + 8


def _pick_block(d, rows, y, W, M, title=None):
    """색 이름 + 이유 목록"""
    if title:
        d.text((M, y), title, font=_font(30, True), fill=_INK)
        y += 48
    nf, rf = _font(31, True), _font(27)
    for r in rows:
        d.rounded_rectangle([M, y + 4, M + 26, y + 30], radius=8, fill=_hex_rgb(r["hex"]))
        d.text((M + 40, y), r["name"], font=nf, fill=_INK)
        y += 44
        # why_short 는 목록 안에서 중복을 걸러낸 것 — 같은 문장이 반복되지 않는다
        reasons = r.get("why_short") or (r.get("why") or [])[:1]
        for w in reasons[:2]:
            for ln in _wrap(d, "· " + w, rf, W - M * 2 - 40):
                d.text((M + 40, y), ln, font=rf, fill=_SOFT)
                y += 36
        y += 14
    return y


def _table(d, pairs, y, W, M):
    kf, vf = _font(27), _font(28)
    for k, v in pairs:
        d.text((M, y + 2), k, font=kf, fill=_SOFT)
        lines = _wrap(d, v, vf, W - M * 2 - 150)
        for i, ln in enumerate(lines):
            d.text((M + 150, y + i * 38), ln, font=vf, fill=_INK)
        y += max(38 * len(lines), 38) + 14
        d.line([(M, y - 8), (W - M, y - 8)], fill=_LINE, width=1)
    return y + 10


def make_page_colors(r: dict, width: int = 1080, as_img: bool = False):
    """2장: 잘 받는 색 / 피하면 좋은 색"""
    W, H, M = width, 2600, 72
    img, d = _page_bg(W, H)
    accent = _hex_rgb(r["percentages"][0]["color"]) if r["percentages"] else (242, 121, 157)

    y = _page_head(d, W, f'{r["info"]["label"]} · 색 가이드', r["info"]["likeness"], accent)
    d.text((M, y), "나에게 잘 받는 색", font=_font(36, True), fill=_INK)
    y = _swatch_grid(d, r["info"]["palette"], M, y + 56, W, M)
    y = _pick_block(d, r.get("best") or [], y + 20, W, M)

    y += 16
    d.line([(M, y), (W - M, y)], fill=_LINE, width=2)
    y += 36
    d.text((M, y), "피하면 좋은 색", font=_font(36, True), fill=_INK)
    y += 50
    d.text((M, y), "안 어울린다기보다, 얼굴이 손해 보는 색입니다.", font=_font(27), fill=_SOFT)
    y = _swatch_grid(d, r["info"]["avoid"], M, y + 46, W, M)
    y = _pick_block(d, r.get("worst") or [], y + 20, W, M, "내 측정값 기준으로 가장 안 맞는 색")

    out = _page_foot(img, d, W, int(y + 110), r["info"]["label"], 2, 3)
    return out if as_img else _to_png(out)


def make_page_style(r: dict, width: int = 1080, as_img: bool = False):
    """3장: 메이크업·헤어 / 타입 특징"""
    W, H, M = width, 2200, 72
    img, d = _page_bg(W, H)
    accent = _hex_rgb(r["percentages"][0]["color"]) if r["percentages"] else (242, 121, 157)

    y = _page_head(d, W, f'{r["info"]["label"]} · 스타일', r["info"]["image"], accent)
    d.text((M, y), "메이크업 · 헤어", font=_font(36, True), fill=_INK)
    y = _table(d, list(r["info"]["makeup"].items()), y + 56, W, M)

    y += 30
    d.text((M, y), f'{r["info"]["label"]} 의 특징', font=_font(36, True), fill=_INK)
    pairs = list(r["info"]["features"].items()) + list(r["info"]["subtypes"])
    y = _table(d, pairs, y + 56, W, M)

    note = "조명과 카메라에 따라 결과가 달라질 수 있습니다. 흰 종이를 함께 촬영하면 더 정확해집니다."
    nf = _font(26)
    lines = _wrap(d, note, nf, W - M * 2 - 56)
    y += 20
    d.rounded_rectangle([M, y, W - M, y + len(lines) * 36 + 44], radius=22,
                        fill=(255, 247, 250), outline=(255, 217, 228), width=2)
    ty = y + 22
    for ln in lines:
        d.text((M + 28, ty), ln, font=nf, fill=_SOFT)
        ty += 36
    y = ty + 30

    out = _page_foot(img, d, W, int(y + 90), r["info"]["label"], 3, 3)
    return out if as_img else _to_png(out)


def _to_png(img) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_long(r: dict, capture_dir: Path, static_images: Path | None = None,
              width: int = 1080) -> bytes:
    """리포트 전체를 '긴 사진 한 장' 으로.

    브라우저(Playwright) 없이 PIL 만으로 만든다. 세 페이지를 세로로 잇고,
    마지막에 시즌 스타일 가이드 사진까지 붙여 내용이 빠지지 않게 한다.
    """
    parts = [
        make_card(r, capture_dir, width, as_img=True),
        make_page_colors(r, width, as_img=True),
        make_page_style(r, width, as_img=True),
    ]

    # 스타일 가이드(코디·화장품) 사진도 같이
    names = {
        "spring_warm": ["최종_봄.png", "봄_화장품.png", "봄_코디.png"],
        "summer_cool": ["최종_여름.png", "여름_화장품.png", "여름_코디.png"],
        "autumn_warm": ["최종_가을.png", "가을_화장품.png", "가을_코디.png"],
        "winter_cool": ["최종_겨울.png", "겨울_화장품.png", "겨울_코디.png"],
    }.get(r.get("season_key"), [])
    guides = []
    if static_images:
        for n in names:
            f = static_images / n
            if not f.exists():
                continue
            try:
                g = Image.open(f).convert("RGB")
                guides.append(g.resize((width, round(g.height * width / g.width))))
            except Exception:
                pass

    gap = 18
    total = sum(p.height for p in parts) + sum(g.height + gap for g in guides) + gap * len(parts)
    if guides:
        total += 90                                   # 가이드 섹션 제목 자리
    out = Image.new("RGB", (width, total), (255, 247, 250))
    d = ImageDraw.Draw(out)

    y = 0
    for p_img in parts:
        out.paste(p_img, (0, y))
        y += p_img.height + gap
    if guides:
        d.text((width // 2, y + 22), "스타일 가이드", font=_font(38, True),
               fill=_INK, anchor="mt")
        y += 90
        for g in guides:
            out.paste(g, (0, y))
            y += g.height + gap

    return _to_png(out.crop((0, 0, width, min(y + 10, total))))
