"""
=====================================================================
 landmarks.py — [3단계-①] MediaPipe 로 얼굴 478개 점(랜드마크) 찾기
=====================================================================
 하는 일
   사진 → 얼굴 위 478개 점의 픽셀 좌표 (x, y)
     - 0 ~ 467 : 얼굴 윤곽·눈썹·눈·코·입·볼 등 (FaceMesh 기본 468점)
     - 468~477 : 홍채(눈동자) 10점  ← refine/iris 모델이 켜져야 나옴
   점 번호(인덱스)는 MediaPipe가 정한 고정 번호라서
   "205번 점은 항상 오른쪽 볼 근처" 처럼 부위를 번호로 지정할 수 있다.
   → face_color.py 가 이 번호들로 볼·이마·홍채 영역을 만든다.

 MediaPipe 버전이 두 가지라서 둘 다 지원한다 (자동 선택)
   ① tasks 방식 (최신 0.10.30+ / 1.x) : models/face_landmarker.task 파일 필요
   ② solutions 방식 (구버전 ~0.10.21) : 모델이 설치 패키지 안에 들어 있어 파일 불필요
   두 방식 모두 점 번호 체계(478점)가 같으므로, 이후 코드는 어느 쪽이든 똑같이 동작한다.

 확인 도구: python tools/draw_landmarks.py data/samples/photo1.jpg
=====================================================================
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "face_landmarker.task"
NUM_LANDMARKS = 478          # 468(얼굴) + 10(홍채)


class FaceNotFoundError(Exception):
    """사진에서 얼굴을 찾지 못했을 때. 웹에서는 '얼굴이 보이게 다시 찍어주세요' 안내로 연결."""


@dataclass
class FaceLandmarks:
    points: np.ndarray        # (478, 2) 픽셀 좌표 [x, y]  (실수)
    image_size: tuple         # (가로, 세로)
    num_faces: int            # 사진에서 찾은 얼굴 수 (2 이상이면 가장 큰 얼굴을 사용)
    backend: str              # "tasks" 또는 "solutions" (어느 방식으로 찾았는지)

    @property
    def face_box(self) -> tuple[int, int, int, int]:
        """얼굴 점들을 감싸는 사각형 (x1, y1, x2, y2)"""
        x1, y1 = self.points[:468].min(axis=0)
        x2, y2 = self.points[:468].max(axis=0)
        return int(x1), int(y1), int(x2), int(y2)

    @property
    def face_width_ratio(self) -> float:
        """얼굴 가로 폭 / 사진 가로 폭 (얼굴이 너무 작게 찍혔는지 판단용 → 신뢰도 재료)"""
        x1, _, x2, _ = self.face_box
        return (x2 - x1) / self.image_size[0]


# ---------------------------------------------------------------------
# 방식 선택 + 검출기 준비 (한 번 만들어 두고 재사용 → 웹에서 매 요청마다 모델 로딩 X)
# ---------------------------------------------------------------------
_detector_cache: dict = {}


def _pick_backend(model_path: Path) -> str:
    import mediapipe as mp
    if model_path.is_file():
        return "tasks"
    if hasattr(mp, "solutions"):
        return "solutions"
    raise RuntimeError(
        "얼굴 검출 모델 파일이 없습니다.\n"
        f"  {model_path}\n"
        "models/README.md 의 주소에서 face_landmarker.task 를 받아 models 폴더에 넣어주세요."
    )


def _get_detector(backend: str, model_path: Path):
    key = (backend, str(model_path))
    if key in _detector_cache:
        return _detector_cache[key]

    import mediapipe as mp
    if backend == "tasks":
        from mediapipe.tasks.python import BaseOptions, vision
        # 경로 대신 '파일 내용(bytes)'을 넘긴다.
        # → Windows 에서 경로에 한글(예: 시보프)이 있으면 MediaPipe 가 파일을 못 여는 문제 회피
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_path.read_bytes()),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=5,
        )
        det = vision.FaceLandmarker.create_from_options(options)
    else:
        det = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,      # 사진 1장씩 (영상 추적 X)
            max_num_faces=5,
            refine_landmarks=True,       # ★ 홍채 10점(468~477) 포함
            min_detection_confidence=0.5,
        )
    _detector_cache[key] = det
    return det


def detect_face_landmarks(img_bgr: np.ndarray, model_path: str | Path | None = None,
                          backend: str | None = None) -> FaceLandmarks:
    """
    사진에서 얼굴 랜드마크 478점을 찾는다. 얼굴이 여러 개면 가장 큰 얼굴을 고른다.

    img_bgr    : BGR 사진 (조명 보정 전/후 상관없음 — 점 위치는 색에 거의 영향 X)
    model_path : face_landmarker.task 경로 (기본: models/face_landmarker.task)
    backend    : "tasks" / "solutions" 강제 지정 (보통 None = 자동)

    얼굴이 없으면 FaceNotFoundError
    """
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    backend = backend or _pick_backend(model_path)
    det = _get_detector(backend, model_path)

    h, w = img_bgr.shape[:2]
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)      # MediaPipe 는 RGB 순서를 받는다

    if backend == "tasks":
        import mediapipe as mp
        result = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        faces = [[(p.x, p.y) for p in face] for face in result.face_landmarks]
    else:
        result = det.process(rgb)
        faces = [[(p.x, p.y) for p in face.landmark] for face in (result.multi_face_landmarks or [])]

    faces = [f for f in faces if len(f) >= NUM_LANDMARKS]
    if not faces:
        raise FaceNotFoundError("사진에서 얼굴을 찾지 못했습니다. 얼굴이 정면으로 잘 보이게 촬영해주세요.")

    # MediaPipe 좌표는 0~1 비율 → 사진 픽셀 좌표로 변환
    all_pts = [np.array(f[:NUM_LANDMARKS], dtype=np.float32) * np.array([w, h], dtype=np.float32) for f in faces]
    # 가장 큰 얼굴(가로 폭 기준) 선택
    widths = [p[:468, 0].max() - p[:468, 0].min() for p in all_pts]
    best = all_pts[int(np.argmax(widths))]

    return FaceLandmarks(points=best, image_size=(w, h), num_faces=len(faces), backend=backend)
