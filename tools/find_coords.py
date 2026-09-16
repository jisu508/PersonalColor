"""
=====================================================================
 find_coords.py — 좌표 찾기 도구 (기존 find_coords.py 개선판)
=====================================================================
 사진에서 흰 종이·물건 위치를 마우스로 클릭해 좌표를 알아낸다.
 3단계(MediaPipe 자동 검출) 이후엔 얼굴 좌표는 필요 없어지지만,
 흰 종이 좌표 확인·디버깅용으로 계속 쓴다.

 달라진 점:
   - 클릭 지점 중심 80x80 박스를 'x1,y1,x2,y2' 형태로 바로 출력
     → check_lighting.py 의 --white / --obj 에 그대로 복사해 붙이면 됨
   - 한글 경로 사진도 열림 (backend.image_io 사용)

 실행:  python tools/find_coords.py data/samples/photo1.jpg  [--size 80]
        클릭할 때마다 출력 / 아무 키나 누르면 종료
=====================================================================
"""
import argparse
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.image_io import imread  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("photo", help="사진 경로")
ap.add_argument("--size", type=int, default=80, help="출력할 박스 한 변 길이(px), 기본 80")
args = ap.parse_args()

img = imread(args.photo)
h, w = img.shape[:2]
# 화면에 맞게 축소해서 보여주되, 좌표는 '원본' 기준으로 되돌려 출력
scale = min(1.0, 900 / w, 900 / h)
disp = cv2.resize(img, (int(w * scale), int(h * scale)))
half = args.size // 2

print("\n[안내] 흰 종이 중앙 → 물건(볼) 중앙 순서로 클릭하세요. 아무 키나 누르면 종료.\n")


def on_click(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    ox, oy = int(x / scale), int(y / scale)                    # 원본 좌표로 환산
    x1, y1 = max(0, ox - half), max(0, oy - half)
    x2, y2 = min(w, ox + half), min(h, oy + half)
    bgr = img[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0).round().astype(int)
    print(f"  클릭 (x={ox}, y={oy})  →  박스 {x1},{y1},{x2},{y2}   평균 BGR={tuple(bgr)}")
    cv2.rectangle(disp, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 0, 255), 2)
    cv2.imshow("click", disp)


cv2.imshow("click", disp)
cv2.setMouseCallback("click", on_click)
cv2.waitKey(0)
cv2.destroyAllWindows()
