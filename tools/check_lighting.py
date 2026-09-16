"""
=====================================================================
 check_lighting.py — 조명 보정 확인 도구  (check2.py 의 새 버전)
=====================================================================
 check2.py 와 하는 일은 같다:
   (1) 사진별 '보정 전(위) / 보정 후(아래)' 비교 이미지 저장
   (2) 물건(또는 피부) 색을 보정 전/후로 측정 → 두 사진에서 가까워지는지 숫자로 확인
 달라진 점:
   - 파일명·좌표를 코드 수정 없이 '명령어 옵션'으로 넘긴다
   - 계산은 backend/vision/lighting.py 를 불러 쓴다 (계산식 중복 X)
   - 사진 1장만 넣어도 동작 (보정 결과 + 품질 경고만 확인)
   - 비교 이미지에 흰 종이(빨강)·물건(초록) 박스를 그려 좌표 실수를 눈으로 확인

 실행 (프로젝트 최상위 폴더에서, 반드시 한 줄로):
   # 사진 2장 비교 (check2.py 와 같은 사용)
   python tools/check_lighting.py --photo data/samples/photo1.jpg --white 645,1817,725,1897 --obj 1116,1827,1196,1907 --photo data/samples/photo3.jpg --white 745,3130,825,3210 --obj 2144,3111,2224,3191

   # 사진 1장만
   python tools/check_lighting.py --photo data/samples/photo1.jpg --white 645,1817,725,1897

   (여러 줄로 나누려면 줄 끝에 PowerShell은 백틱 ` , 맥/리눅스는 \ )
   ※ data/samples/ 에 사진을 먼저 넣어야 함 (GitHub에는 사진이 없음)

 좌표 모를 때:  python tools/find_coords.py data/samples/photo1.jpg
 결과 이미지 :  outputs/lighting_compare.jpg
=====================================================================
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# 'python tools/xxx.py' 로 실행하면 파이썬이 tools/ 폴더만 보기 때문에
# 프로젝트 최상위 폴더를 import 경로에 추가해 backend 를 찾을 수 있게 한다.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.color_utils import clip_box, delta_e, mean_lab  # noqa: E402
from backend.image_io import imread, imwrite                   # noqa: E402
from backend.vision.lighting import white_balance              # noqa: E402


def parse_box(text: str):
    """'645,1817,725,1897' → (645, 1817, 725, 1897)"""
    try:
        vals = tuple(int(v) for v in text.split(","))
        assert len(vals) == 4
        return vals
    except Exception:
        raise argparse.ArgumentTypeError(f"좌표는 x1,y1,x2,y2 형식이어야 합니다: '{text}'")


def draw_boxes(img, white_box, obj_box):
    """확인용으로 박스를 그린 복사본 (원본은 건드리지 않음). 두께는 사진 크기에 비례."""
    out = img.copy()
    t = max(2, img.shape[1] // 300)
    x1, y1, x2, y2 = clip_box(white_box, img.shape)
    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), t)       # 빨강 = 흰 종이
    if obj_box:
        x1, y1, x2, y2 = clip_box(obj_box, img.shape)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), t)   # 초록 = 물건/피부
    return out


def make_panel(before, after, height=400):
    """위=원본, 아래=보정 후 로 세로로 붙인 패널 (check2.py의 stack 과 같음)"""
    def rs(x):
        return cv2.resize(x, (int(x.shape[1] * height / x.shape[0]), height))
    return np.vstack([rs(before), rs(after)])


def main():
    ap = argparse.ArgumentParser(description="흰 종이 기준 조명 보정 확인")
    ap.add_argument("--photo", action="append", required=True, help="사진 경로 (1~2번 반복 가능)")
    ap.add_argument("--white", action="append", required=True, type=parse_box, help="흰 종이 영역 x1,y1,x2,y2")
    ap.add_argument("--obj", action="append", type=parse_box, help="색을 비교할 물건/피부 영역 x1,y1,x2,y2")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "lighting_compare.jpg"), help="결과 이미지 경로")
    args = ap.parse_args()

    n = len(args.photo)
    objs = args.obj or []
    if len(args.white) != n or (objs and len(objs) != n):
        ap.error("--photo 개수만큼 --white (그리고 쓴다면 --obj) 를 같은 순서로 넣어주세요.")

    panels, before_list, after_list = [], [], []
    print("=" * 64)
    for i in range(n):
        img = imread(args.photo[i])
        obj = objs[i] if objs else None
        wb = white_balance(img, args.white[i])

        print(f"[사진{i+1}] {args.photo[i]}")
        print(f"   종이 평균 BGR  : {np.round(wb.paper_bgr, 1)}   → 배율(B,G,R): {np.round(wb.gains, 3)}")
        print(f"   조명 색끼 세기 : {wb.cast_strength:.1f}   (종이 Lab a,b = {wb.paper_lab[1]:.1f}, {wb.paper_lab[2]:.1f})")
        if obj:
            b, a = mean_lab(img, box=obj), mean_lab(wb.image, box=obj)
            before_list.append(b)
            after_list.append(a)
            print(f"   물건색 Lab     : 보정 전 {np.round(b, 1)}  →  보정 후 {np.round(a, 1)}")
        print("   품질 경고      : " + (" / ".join(wb.warnings) if wb.warnings else "없음"))
        print("-" * 64)

        panels.append(make_panel(draw_boxes(img, args.white[i], obj), draw_boxes(wb.image, args.white[i], obj)))

    # 패널들을 좌우로 이어 붙여 저장 (사이에 흰 여백 20px)
    H = min(p.shape[0] for p in panels)
    gap = np.full((H, 20, 3), 255, np.uint8)
    row = [panels[0][:H]]
    for p in panels[1:]:
        row += [gap, p[:H]]
    out_path = imwrite(args.out, np.hstack(row))

    if n == 2 and len(before_list) == 2:
        d_before = delta_e(*before_list)
        d_after = delta_e(*after_list)
        print(f"두 사진의 물건색 차이(ΔE) | 보정 전: {d_before:5.1f}   보정 후: {d_after:5.1f}")
        print("=" * 64)
        if d_after < d_before:
            print("성공: 보정 후 두 사진의 색 차이가 줄었습니다. → 조명이 달라도 비슷한 색으로 측정됨")
        else:
            print("차이가 줄지 않았습니다. 흰 종이 좌표가 정확한지, 종이에 그림자가 없는지 확인하세요.")
    print(f"\n결과 이미지: {out_path}  (위=원본, 아래=보정 후 / 빨강=종이, 초록=물건)")


if __name__ == "__main__":
    main()
