"""
=====================================================================
 draw_landmarks.py — MediaPipe 얼굴 점(478개)이 제대로 잡히는지 눈으로 확인
=====================================================================
 결과 이미지: 얼굴 부분만 잘라 확대한 뒤
   - 초록 점 : 얼굴 점 0~467
   - 빨강 점 : 홍채 점 468~477
   - --numbers 옵션을 주면 점마다 번호를 적어줌 (부위 번호 고를 때 사용)

 실행:
   python tools/draw_landmarks.py data/samples/photo1.jpg
   python tools/draw_landmarks.py data/samples/photo1.jpg --numbers
 결과: outputs/landmarks_photo1.jpg
=====================================================================
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.image_io import imread, imwrite                     # noqa: E402
from backend.vision.landmarks import detect_face_landmarks       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("photo")
ap.add_argument("--numbers", action="store_true", help="점 번호 표시")
ap.add_argument("--size", type=int, default=1400, help="결과 이미지 한 변 크기(px)")
args = ap.parse_args()

img = imread(args.photo)
lm = detect_face_landmarks(img)
print(f"검출 방식: {lm.backend} / 찾은 얼굴 수: {lm.num_faces} / 얼굴 박스: {lm.face_box} / 얼굴 폭 비율: {lm.face_width_ratio:.2f}")

# 얼굴 주변을 여유 있게 잘라서 확대 (점이 잘 보이도록)
x1, y1, x2, y2 = lm.face_box
pad = int((x2 - x1) * 0.25)
x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
x2, y2 = min(img.shape[1], x2 + pad), min(img.shape[0], y2 + pad)
crop = img[y1:y2, x1:x2]
scale = args.size / max(crop.shape[:2])
out = cv2.resize(crop, None, fx=scale, fy=scale)

for i, (x, y) in enumerate(lm.points):
    px, py = int((x - x1) * scale), int((y - y1) * scale)
    color = (0, 0, 255) if i >= 468 else (0, 255, 0)
    cv2.circle(out, (px, py), 2, color, -1)
    if args.numbers:
        cv2.putText(out, str(i), (px + 2, py - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1)

name = Path(args.photo).stem
path = imwrite(ROOT / "outputs" / f"landmarks_{name}{'_numbers' if args.numbers else ''}.jpg", out)
print(f"결과 이미지: {path}")
