"""
=====================================================================
 check_face_color.py — 얼굴 부위별 색 자동 추출 확인 도구 (3단계)
=====================================================================
 하는 일
   (1) 흰 종이로 조명 보정 → MediaPipe 로 볼·이마·홍채 자동 검출 → 부위별 Lab 출력
   (2) 어느 영역을 썼는지 그린 이미지 저장 (초록=사용 / 빨강=제외 / 하늘색=눈꺼풀 윤곽)
   (3) 사진 2장을 넣으면: 피부·눈 색이 두 사진에서 가까워지는지 보정 전/후 비교
       → check_lighting.py 와 같은 실험을, 이번엔 '좌표 클릭 없이' 한다.

 실행 (프로젝트 최상위 폴더에서, 한 줄로):
   python tools/check_face_color.py --photo data/samples/photo1.jpg --white 645,1817,725,1897 --photo data/samples/photo3.jpg --white 745,3130,825,3210

   # 흰 종이 없이 (조명 보정 생략, 추출만 확인)
   python tools/check_face_color.py --photo data/samples/photo1.jpg

 결과 이미지: outputs/face_color_<사진이름>.jpg
=====================================================================
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.color_utils import delta_e                                    # noqa: E402
from backend.image_io import imread, imwrite                               # noqa: E402
from backend.vision.face_color import draw_regions, extract_face_colors    # noqa: E402
from backend.vision.lighting import white_balance                          # noqa: E402

LABELS = {"cheek_right": "오른쪽 볼", "cheek_left": "왼쪽 볼", "forehead": "이마",
          "iris_a": "홍채 A", "iris_b": "홍채 B"}


def parse_box(text):
    vals = tuple(int(v) for v in text.split(","))
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("x1,y1,x2,y2 형식")
    return vals


def fmt(lab):
    return "없음" if lab is None else f"L {lab[0]:5.1f}  a {lab[1]:5.1f}  b {lab[2]:5.1f}"


def main():
    ap = argparse.ArgumentParser(description="얼굴 부위별 색 자동 추출 확인")
    ap.add_argument("--photo", action="append", required=True)
    ap.add_argument("--white", action="append", type=parse_box, help="흰 종이 영역 (생략하면 보정 없이)")
    args = ap.parse_args()
    whites = args.white or []
    if whites and len(whites) != len(args.photo):
        ap.error("--white 를 쓰려면 --photo 개수만큼 같은 순서로 넣어주세요.")

    results = []   # (보정 전 결과, 보정 후 결과)
    for i, path in enumerate(args.photo):
        img = imread(path)
        before = extract_face_colors(img)                      # 랜드마크는 여기서 한 번만 찾고
        after = None
        if whites:
            wb = white_balance(img, whites[i])
            after = extract_face_colors(wb.image, landmarks=before.landmarks)   # 보정본에 재사용
        shown = after or before
        lm = shown.landmarks

        print("=" * 66)
        print(f"[사진{i+1}] {path}")
        print(f"   얼굴 검출 : {lm.backend} 방식 / 얼굴 {lm.num_faces}개 / 얼굴 폭 {lm.face_width_ratio:.0%}")
        print(f"   {'부위':<8}{'사용':<5}{'픽셀(사용/전체)':<18}Lab ({'보정 후' if after else '보정 없음'})")
        for key, r in shown.regions.items():
            mark = "O" if r.used else "X"
            print(f"   {LABELS[key]:<8}{mark:<5}{f'{r.n_used}/{r.n_total}':<18}{fmt(r.lab)}  {r.note}")
        if after:
            print(f"   ▶ 피부 : 보정 전 {fmt(before.skin)}  →  보정 후 {fmt(after.skin)}")
            print(f"   ▶ 눈   : 보정 전 {fmt(before.eye)}  →  보정 후 {fmt(after.eye)}")
        else:
            print(f"   ▶ 피부 : {fmt(before.skin)}")
            print(f"   ▶ 눈   : {fmt(before.eye)}")
        print("   경고   : " + (" / ".join(shown.warnings) if shown.warnings else "없음"))

        # 영역 그림: 얼굴 주변만 잘라 저장
        vis = draw_regions(wb.image if after else img, shown)
        x1, y1, x2, y2 = lm.face_box
        pad = (x2 - x1) // 5
        crop = vis[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        crop = cv2.resize(crop, None, fx=900 / crop.shape[0], fy=900 / crop.shape[0])
        out = imwrite(ROOT / "outputs" / f"face_color_{Path(path).stem}.jpg", crop)
        print(f"   영역 이미지: {out}")
        results.append((before, after))

    if len(results) == 2 and whites:
        print("=" * 66)
        print("두 사진 비교 (ΔE: 전체 / a·b 색만)")
        for part, name in (("skin", "피부"), ("eye", "눈")):
            b1, b2 = getattr(results[0][0], part), getattr(results[1][0], part)
            a1, a2 = getattr(results[0][1], part), getattr(results[1][1], part)
            if any(v is None for v in (b1, b2, a1, a2)):
                print(f"   {name}: 추출 실패로 비교 불가")
                continue
            ab = lambda x, y: float(np.linalg.norm((x - y)[1:]))   # noqa: E731
            print(f"   {name}: 보정 전 {delta_e(b1, b2):5.1f} / {ab(b1, b2):5.1f}   →   "
                  f"보정 후 {delta_e(a1, a2):5.1f} / {ab(a1, a2):5.1f}")


if __name__ == "__main__":
    main()
