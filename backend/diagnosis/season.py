"""
=====================================================================
 season.py — [4단계] 시즌 판정 (웜/쿨 → 4계절 → 퍼센티지)
=====================================================================
 입력 : 얼굴에서 뽑은 Lab 색 (backend.vision.face_color 결과)
 출력 : 시즌별 퍼센티지 + 근거 숫자

 판정은 두 단계다.
   1단계 웜/쿨  : 피부의 b(노란 정도)를 기준값과 비교        ← 핵심
   2단계 4계절  : 밝기(L)와 채도(C)로 봄/가을, 여름/겨울 구분  ← 아직 검증 전(임시)

 웜/쿨 판정 방법 4가지를 넣어두고 골라 쓸 수 있게 했다.
   "bc"       D = b - C                 ← 기본값. 우리 팀이 모은 연예인 40명에서 정확도 100%
                                           (b(노란기)와 C(채도)를 같이 보는 값. 웜은 둘 다 높다)
   "b"        피부 b값만 사용            ← 홍조(a)에 안 흔들림. 40명에서 92%
   "paper_d"  논문① 판별식 D             ← 선행연구. 40명에서 98%
   "smtc"     논문③·ShowMeTheColor 방식  ← 선행연구 비교용
 세 방법을 같은 사진에 돌려 비교하려면 compare_methods() 를 쓴다.

 기준값은 코드가 아니라 data/reference/season_reference.json 에 있다.
 (지금 값은 임시. 라벨 붙은 사진이 모이면 tools/fit_threshold.py 로 다시 계산)
=====================================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_PATH = ROOT / "data" / "reference" / "season_reference.json"

_cache: dict = {}


def load_reference(path: str | Path | None = None) -> dict:
    """기준값 JSON 읽기 (한 번 읽으면 기억해 둔다)"""
    path = Path(path) if path else REFERENCE_PATH
    key = str(path)
    if key not in _cache:
        _cache[key] = json.loads(path.read_text(encoding="utf-8"))
    return _cache[key]


@dataclass
class SeasonResult:
    seasons: dict            # {"spring_warm": 12.0, ...} 합계 100
    top: str                 # 1위 시즌 키
    runner_up: str           # 2위 시즌 키
    gap: float               # 1위 - 2위 (신뢰도 계산에 사용)
    warm_score: float        # 0보다 크면 웜 쪽 (경계에서 얼마나 떨어졌는지)
    warm_cool: str           # "warm" 또는 "cool"
    method: str              # 쓴 판정 방법
    detail: dict = field(default_factory=dict)   # 근거 숫자들

    def to_dict(self) -> dict:
        return {"seasons": {k: round(v, 1) for k, v in self.seasons.items()},
                "top": self.top, "runner_up": self.runner_up, "gap": round(self.gap, 1),
                "warm_cool": self.warm_cool, "warm_score": round(self.warm_score, 2),
                "method": self.method, "detail": self.detail}


# ---------------------------------------------------------------------
# 웜/쿨 점수 — 세 가지 방법
#   반환값: (점수, 근거dict)  점수 > 0 이면 웜, < 0 이면 쿨, 0에 가까우면 경계
# ---------------------------------------------------------------------
def _score_bc(colors: dict, ref: dict):
    """
    D = b - C   (C = √(a²+b²))
    값은 항상 0 이하이고, 0에 가까울수록 웜 쪽이다.
    피부가 노랗고(b 큼) 색이 진할수록(C 큼) 웜으로 가는 성질이 있어,
    우리 팀이 모은 연예인 40명(웜20/쿨20)에서 이 값이 가장 잘 갈렸다(100%).
    """
    cfg = ref["warm_cool"]["bc"]
    lab = colors["skin"]
    a, b = float(lab[1]), float(lab[2])
    C = float(np.hypot(a, b))
    D = b - C
    return (D - cfg["threshold"]) / cfg["scale"], {"D_bc": round(D, 2), "threshold": cfg["threshold"],
                                                  "skin_a": round(a, 1), "skin_b": round(b, 1), "skin_C": round(C, 1)}


def _score_b(colors: dict, ref: dict):
    """피부 b값만 사용 (기본)"""
    cfg = ref["warm_cool"]["b"]
    b = float(colors["skin"][2])
    return (b - cfg["threshold"]) / cfg["scale"], {"skin_b": round(b, 1), "threshold": cfg["threshold"]}


def _score_paper_d(colors: dict, ref: dict):
    """논문① 판별식:  D = const + a계수*a + b계수*b,  D > 0.463 이면 웜"""
    cfg = ref["warm_cool"]["paper_d"]
    c = cfg["coef"]
    a, b = float(colors["skin"][1]), float(colors["skin"][2])
    D = c["const"] + c["a"] * a + c["b"] * b
    return (D - cfg["threshold"]) / cfg["scale"], {"D": round(D, 2), "threshold": cfg["threshold"],
                                                   "skin_a": round(a, 1), "skin_b": round(b, 1)}


def _score_smtc(colors: dict, ref: dict):
    """논문③·ShowMeTheColor: 피부·머리·눈의 b값을 웜/쿨 기준값과 비교해 가까운 쪽"""
    cfg = ref["warm_cool"]["smtc"]
    parts = ["skin", "hair", "eye"]
    warm_d = cool_d = 0.0
    used = []
    for i, p in enumerate(parts):
        lab = colors.get(p)
        if lab is None:                     # 머리·눈 추출 실패 시 그 부위는 건너뛴다
            continue
        b = float(lab[2])
        warm_d += abs(b - cfg["warm_std"][i]) * cfg["weights"][i]
        cool_d += abs(b - cfg["cool_std"][i]) * cfg["weights"][i]
        used.append(p)
    if not used:
        raise ValueError("피부·머리·눈 색이 하나도 없어 판정할 수 없습니다.")
    # 쿨거리가 멀수록(=웜에 가까울수록) 점수가 커지도록
    return (cool_d - warm_d) / cfg["scale"], {"warm_distance": round(warm_d, 1),
                                              "cool_distance": round(cool_d, 1), "used_parts": used}


_METHODS = {"bc": _score_bc, "b": _score_b, "paper_d": _score_paper_d, "smtc": _score_smtc}


def warm_cool_score(colors: dict, ref: dict | None = None, method: str | None = None):
    """웜/쿨 점수 하나를 계산한다. 양수면 웜, 음수면 쿨."""
    ref = ref or load_reference()
    method = method or ref["warm_cool"]["method"]
    if method not in _METHODS:
        raise ValueError(f"모르는 판정 방법: {method} (가능: {list(_METHODS)})")
    return _METHODS[method](colors, ref)


# ---------------------------------------------------------------------
# 4계절 퍼센티지
# ---------------------------------------------------------------------
def diagnose_season(colors: dict, ref: dict | None = None, method: str | None = None) -> SeasonResult:
    """
    ★ 보통 이 함수만 쓰면 된다 ★
    부위별 Lab → 시즌별 퍼센티지

    colors 예) {"skin": [74.4, 15.2, 13.3], "eye": [19.5, 6.7, 12.8], "hair": [25.2, 3.5, 5.3]}
              (hair, eye 는 None 이어도 된다 — 방법 "b" 는 피부만 쓴다)

    계산 방법
      1) 웜/쿨 점수, 밝기 점수, 채도 점수 세 개를 구한다 (각각 0 기준으로 ±)
      2) 시즌 4개는 이 세 축에서 각각 +1/-1 위치를 갖는다 (JSON에 정의)
      3) 내 점수와 각 시즌 위치의 거리를 재고, softmax 로 퍼센티지로 바꾼다
    """
    ref = ref or load_reference()
    method = method or ref["warm_cool"]["method"]
    skin = colors.get("skin")
    if skin is None:
        raise ValueError("피부색(skin)이 없어 판정할 수 없습니다.")

    warm, detail = warm_cool_score(colors, ref, method)

    L = float(skin[0])
    C = float(np.hypot(skin[1], skin[2]))
    axis = ref["second_axis"]
    light = (L - axis["lightness"]["center"]) / axis["lightness"]["scale"]
    chroma = (C - axis["chroma"]["center"]) / axis["chroma"]["scale"]

    # 점수가 너무 크면 한쪽으로만 쏠리므로 -2 ~ +2 로 제한
    v = np.clip([warm, light, chroma], -2.0, 2.0)

    names, dists = [], []
    for key, s in ref["seasons"].items():
        center = np.array([s["warm"], s["light"], s["chroma"]], dtype=float)
        # 웜/쿨 축에 2배 비중 (웜/쿨이 먼저 갈리고, 나머지는 보조)
        weight = np.array([2.0, 1.0, 1.0])
        names.append(key)
        dists.append(float(np.sqrt((weight * (v - center) ** 2).sum())))

    d = np.array(dists)
    logits = -d / max(1e-6, ref.get("softmax_temperature", 1.2))
    p = np.exp(logits - logits.max())
    p = 100.0 * p / p.sum()

    order = np.argsort(-p)
    seasons = {names[i]: float(p[i]) for i in range(len(names))}
    detail.update({"skin_L": round(L, 1), "skin_C": round(C, 1),
                   "axis": {"warm": round(float(v[0]), 2), "light": round(float(v[1]), 2),
                            "chroma": round(float(v[2]), 2)},
                   "provisional_second_axis": True})

    return SeasonResult(seasons=seasons, top=names[order[0]], runner_up=names[order[1]],
                        gap=float(p[order[0]] - p[order[1]]), warm_score=float(warm),
                        warm_cool="warm" if warm > 0 else "cool", method=method, detail=detail)


def compare_methods(colors: dict, ref: dict | None = None) -> dict:
    """세 가지 판정 방법을 같은 색에 모두 돌려 비교한다. (어느 방법을 쓸지 정할 때)"""
    ref = ref or load_reference()
    out = {}
    for m in _METHODS:
        try:
            score, detail = warm_cool_score(colors, ref, m)
            out[m] = {"warm_cool": "warm" if score > 0 else "cool", "score": round(score, 2), **detail}
        except Exception as e:
            out[m] = {"error": f"{type(e).__name__}: {e}"}
    return out


def season_label(key: str, ref: dict | None = None) -> str:
    """영문 키 → 화면에 보여줄 한글 이름"""
    ref = ref or load_reference()
    return ref["seasons"].get(key, {}).get("label", key)


def season_color(key: str, ref: dict | None = None) -> str:
    """영문 키 → 화면 표시용 색상 코드"""
    ref = ref or load_reference()
    return ref["seasons"].get(key, {}).get("color", "#888888")
