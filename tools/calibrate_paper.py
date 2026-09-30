r"""
=====================================================================
 calibrate_paper.py — 흰 종이가 진짜 '중성'인지 재고, 치우침을 보정값으로 저장
=====================================================================
 왜 필요한가
   복사용지에는 형광증백제(종이를 더 하얗게 보이게 하는 물질)가 들어 있어서
   조명에 따라 사진에 파랗게 찍힙니다. 그 종이를 '완전한 흰색'으로 맞추면
   보정이 과해져서 피부가 통째로 웜 쪽으로 밀립니다. 실측으로 h 가 +15~22° 움직였습니다.

 어떻게 재나
   한 장의 사진 안에 ① 흰 종이 와 ② 진짜 중성인 기준물 을 나란히 두고 찍습니다.
   기준물로 쓸 수 있는 것: 회색 카드(제일 정확), 무광 흰 타일, 형광증백제 없는 백색 PVC 카드.
   두 영역의 색 비율이 곧 종이의 치우침입니다.

 실행
   # 좌표를 모르면 먼저 확인
   python tools\find_coords.py 사진.jpg

   python tools\calibrate_paper.py 사진.jpg --paper 767,129,1254,662 --neutral 100,80,300,240
   python tools\calibrate_paper.py 사진.jpg --paper ... --neutral ... --write

 --write 를 붙이면 data/reference/season_reference.json 의
 white_balance.reference_tint 에 저장되고, 이후 모든 진단에 자동 적용됩니다.
=====================================================================
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.image_io import imread                                    # noqa: E402

REF_PATH = ROOT / "data/reference/season_reference.json"


def box(s: str):
    v = [int(x) for x in s.split(",")]
    if len(v) != 4:
        raise argparse.ArgumentTypeError("x1,y1,x2,y2 형식으로 넣어주세요")
    return v


def mean_clean(img, b):
    """포화(250 이상) 화소를 뺀 평균 BGR"""
    x1, y1, x2, y2 = b
    px = img[y1:y2, x1:x2].reshape(-1, 3).astype(np.float32)
    keep = px[(px < 250).all(axis=1)]
    if len(keep) < 50:
        raise SystemExit(f"영역 {b} 에 쓸 만한 화소가 없습니다 (대부분 하얗게 날아감)")
    return keep.mean(axis=0)


def main():
    ap = argparse.ArgumentParser(description="흰 종이의 색 치우침 측정")
    ap.add_argument("image")
    ap.add_argument("--paper", type=box, required=True, help="흰 종이 영역 x1,y1,x2,y2")
    ap.add_argument("--neutral", type=box, required=True, help="진짜 중성인 기준물 영역 x1,y1,x2,y2")
    ap.add_argument("--write", action="store_true", help="결과를 season_reference.json 에 저장")
    a = ap.parse_args()

    img = imread(a.image)
    p, n = mean_clean(img, a.paper), mean_clean(img, a.neutral)

    # 종이가 중성물 대비 어느 쪽으로 치우쳤는지 (밝기 차이는 빼고 색만 본다)
    ratio = (p / p.mean()) / (n / n.mean())
    tint = [float(x) for x in (ratio / ratio.mean())]

    print(f"흰 종이  BGR {p.round(1).tolist()}   B-R {p[0]-p[2]:+.1f}")
    print(f"기준물   BGR {n.round(1).tolist()}   B-R {n[0]-n[2]:+.1f}")
    print(f"\n종이 치우침 (reference_tint) [B, G, R] = {[round(x, 4) for x in tint]}")
    dev = max(abs(x - 1) for x in tint) * 100
    print(f"중성에서 벗어난 정도: 최대 {dev:.1f}%")
    if dev < 2:
        print("  → 이 종이는 거의 중성입니다. 보정 없이 그대로 써도 됩니다.")
    elif tint[0] > 1.02:
        print("  → 종이가 파랗게 찍힙니다(형광증백제 의심). 보정하지 않으면 피부가 웜 쪽으로 밀립니다.")
    elif tint[2] > 1.02:
        print("  → 종이가 누렇게 찍힙니다. 보정하지 않으면 피부가 쿨 쪽으로 밀립니다.")

    if a.write:
        d = json.loads(REF_PATH.read_text(encoding="utf-8"))
        before = d["white_balance"].get("reference_tint")
        d["white_balance"]["reference_tint"] = [round(x, 4) for x in tint]
        d["white_balance"]["_reference_tint_측정"] = (
            f"{Path(a.image).name} 에서 측정. 종이 {p.round(0).astype(int).tolist()} / "
            f"기준물 {n.round(0).astype(int).tolist()}")
        REF_PATH.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n저장했습니다: reference_tint {before} → {[round(x, 4) for x in tint]}")
        print("같은 종이·같은 조명에서만 유효합니다. 종이를 바꾸면 다시 재세요.")


if __name__ == "__main__":
    main()
