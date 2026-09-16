# web — [7단계] Flask 서버 + 웹캠 촬영 화면 (담당: 웹)

예정 구성
- `app.py` : Flask 서버. 사진 받기 → `backend.pipeline.run_pipeline()` 호출 → JSON 응답
- `templates/index.html` : 촬영 화면 (getUserMedia + 흰 종이 가이드 오버레이)
- `static/js/`, `static/css/`

규칙: web/ 안에서는 영상처리·판정 계산을 하지 않는다. 계산은 전부 backend/ 에서.
참고: getUserMedia는 `localhost` 또는 HTTPS에서만 카메라가 켜진다.
