# models — MediaPipe 모델 파일

| 파일 | 용도 | 받는 곳 |
|---|---|---|
| `face_landmarker.task` | 얼굴 478개 점(홍채 포함) 검출 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task |

- 브라우저로 위 주소를 열어 받은 파일을 이 폴더에 그대로 넣으면 된다.
- 약 3.6MB, Apache 2.0 라이선스 (Google MediaPipe). 인터넷 없이 로컬에서 실행된다.
- 이 파일이 없고 MediaPipe 가 구버전(0.10.21 이하, `mp.solutions` 있음)이면 구버전 방식으로 자동 동작한다.
