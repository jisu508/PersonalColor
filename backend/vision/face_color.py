"""
[3단계 — 아직 미구현] MediaPipe FaceMesh(홍채 포함)로 얼굴 자동 검출 → 부위별 평균 Lab

담당: 영상처리 파트
예정 인터페이스 (팀 합의용 초안 — 구현 시 바뀔 수 있음):

    extract_face_colors(img_bgr) -> dict
        {
          "skin":  [L, a, b],   # 양 볼 + 이마 평균
          "eye":   [L, a, b],   # 홍채
          "hair":  [L, a, b],   # 이마 위쪽 머리 영역
          "face_box": (x1, y1, x2, y2),
        }

입력은 반드시 lighting.white_balance() 로 '보정된' 사진이어야 한다.
색 평균은 backend.color_utils.mean_lab(img, mask=...) 를 사용할 것.
"""


def extract_face_colors(img_bgr):
    raise NotImplementedError("3단계에서 구현 예정")
