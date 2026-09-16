"""
image_io.py — 사진 읽기/저장 (공용)

왜 cv2.imread 대신 이걸 쓰나?
  Windows에서 cv2.imread / cv2.imwrite 는 경로에 '한글'이 있으면
  에러 없이 실패(None 반환 / 파일 안 생김)한다.
  예) C:\\Users\\user\\Desktop\\시보프\\photo1.jpg  ← '시보프' 때문에 실패 가능
  NumPy로 파일을 바이트로 읽은 뒤 OpenCV로 해독하면 한글 경로도 문제없다.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: str | Path) -> np.ndarray:
    """사진을 BGR uint8 배열로 읽는다. 실패하면 None 대신 에러를 낸다(원인 찾기 쉬움)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"사진 파일이 없습니다: {path}")
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"사진으로 읽을 수 없는 파일입니다: {path}")
    return img


def imwrite(path: str | Path, img: np.ndarray, quality: int = 92) -> Path:
    """사진 저장. 폴더가 없으면 만든다. 확장자(.jpg/.png)로 형식 결정."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"이미지 인코딩 실패: {path}")
    buf.tofile(str(path))
    return path
