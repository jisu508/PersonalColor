"""
Flask 서버 — 웹캠 촬영 화면 + 진단 API

  /            촬영 화면 (templates/index.html)
  /api/diagnose  사진 + 흰 종이 가이드 박스 좌표 → 진단 결과 JSON

진단 계산은 전부 backend/ 안에 있고, 여기서는 run_pipeline() 하나만 부른다.
자세한 결과 형식은 docs/web_integration.md 참고.

실행:  python app.py   →  http://127.0.0.1:5000
"""
import json

from flask import Flask, jsonify, render_template, request

from backend.image_io import imdecode_bytes
from backend.pipeline import run_pipeline

app = Flask(__name__)


# 1. 메인 화면(HTML) 띄우기
@app.route('/')
def home():
    return render_template('index.html')


# 2. 사진 캡처 및 진단 API
@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    # 프론트엔드에서 보낸 이미지 파일 받기
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    try:
        img = imdecode_bytes(request.files['image'].read())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # 프론트엔드에서 보낸 흰 종이 가이드 박스 좌표 (비율 0~1 또는 픽셀)
    # 좌표가 없으면 백엔드가 사진에서 흰 종이를 자동으로 찾는다.
    coords_json = request.form.get('coords')
    guide_box = None
    if coords_json:
        try:
            guide_box = json.loads(coords_json)
        except json.JSONDecodeError:
            guide_box = None

    # 진단 실행 (조명 보정 → 얼굴 색 추출 → 시즌 판정 → 신뢰도)
    result = run_pipeline(img, guide_box=guide_box)

    if not result["ok"]:
        # 흰 종이를 못 찾음 / 얼굴을 못 찾음 등 — message 를 화면에 그대로 보여주면 된다
        return jsonify({"error": result["message"], "code": result["error"]}), 400

    return jsonify(result)


if __name__ == '__main__':
    # debug=True 로 두면 코드를 수정할 때마다 서버가 자동 재시작된다
    app.run(debug=True)
