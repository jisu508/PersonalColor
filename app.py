"""
Flask 서버 — 웹캠 촬영 화면 + 진단 API

  GET  /               촬영 화면 (templates/index.html)
  POST /api/diagnose   사진(+흰 종이 가이드 박스 좌표) → 진단 결과 JSON

진단 계산은 전부 backend/ 안에 있고, 여기서는 run_pipeline() 하나만 부른다.
결과 형식은 docs/web_integration.md 참고.

실행:  python app.py   →  http://127.0.0.1:5000
"""
import base64
import json
import traceback

import cv2
from flask import Flask, jsonify, render_template, request

from backend.image_io import imdecode_bytes
from backend.pipeline import run_pipeline

app = Flask(__name__)


def _jsonable(obj):
    """numpy 숫자·배열이 섞여 있으면 jsonify 가 터진다 → 순수 파이썬 값으로 바꿔준다"""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _valid_box(box):
    """
    가이드 박스 좌표 검사. 이상하면 None 을 돌려 자동 탐지로 넘어간다.

    프론트에서 숨겨진 요소의 크기를 재면 계산이 NaN 이 되고, JSON 으로 오면 null 이 된다.
    그대로 쓰면 서버가 터지므로 여기서 걸러낸다.
    """
    import math
    if not isinstance(box, dict) or not box:
        return None
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
             for v in box.values())
    if not ok:
        app.logger.warning("가이드 박스 좌표가 올바르지 않아 무시합니다(자동 탐지로 진행): %r", box)
        return None
    return box


def _to_data_url(img_bgr) -> str | None:
    """BGR 배열 → <img src="..."> 에 바로 넣을 수 있는 data URL (JPEG)"""
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii")


# 1. 메인 화면(HTML) 띄우기
@app.route('/')
def home():
    return render_template('index.html')


# 1-1. 브라우저가 자동으로 찾는 아이콘 — 없으면 콘솔에 빨간 404 가 뜨므로 빈 응답을 준다
@app.route('/favicon.ico')
def favicon():
    return '', 204


# 2-0. 어떤 오류가 나도 HTML 대신 JSON 으로 돌려준다.
#      (기본 Flask 는 오류 때 HTML 페이지를 주는데, 프론트가 JSON.parse 하다 터진다)
@app.errorhandler(Exception)
def handle_any_error(e):
    traceback.print_exc()                     # 터미널에는 전체 오류 내용을 그대로 남긴다
    code = getattr(e, "code", 500)
    return jsonify({
        "ok": False,
        "code": "server_error",
        "error": f"서버 내부 오류: {type(e).__name__}: {e}",
    }), code if isinstance(code, int) else 500


# 2. 사진 캡처 및 진단 API
@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    # 프론트엔드에서 보낸 이미지 파일 받기
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다.", "code": "no_image"}), 400

    try:
        img = imdecode_bytes(request.files['image'].read())
    except ValueError as e:
        return jsonify({"error": str(e), "code": "bad_image"}), 400

    # 흰 종이 가이드 박스 좌표 (0~1 비율 또는 픽셀).
    #   {"x":..,"y":..,"width":..,"height":..} 와 {"x1":..,..} 둘 다 받는다.
    #   좌표가 없으면 백엔드가 사진에서 흰 종이를 자동으로 찾는다.
    coords_json = request.form.get('coords')
    guide_box = None
    if coords_json:
        try:
            guide_box = _valid_box(json.loads(coords_json))
        except json.JSONDecodeError:
            guide_box = None

    # 진단 실행 (조명 보정 → 얼굴 색 추출 → 웜/쿨·시즌 판정 → 신뢰도)
    result = run_pipeline(img, guide_box=guide_box, return_images=True)

    # 보정된 사진을 화면에 보여줄 수 있게 data URL 로 바꿔 넣는다 ('근거를 보여주는' 부분)
    images = result.pop("images", None) or {}
    result = _jsonable(result)          # numpy 숫자를 파이썬 숫자로 (jsonify 가 numpy 를 못 다룬다)
    result["corrected_image"] = _to_data_url(images["corrected"]) if "corrected" in images else None

    if not result["ok"]:
        # 촬영 조건 미달 / 얼굴 못 찾음 등 — error 를 화면에 그대로 보여주면 된다
        return jsonify({
            "ok": False,
            "code": result["error"],
            "error": result["message"],
            "reasons": result.get("reasons", []),
            "corrected_image": result["corrected_image"],
        }), 400

    return jsonify(result)


if __name__ == '__main__':
    # debug=True 로 두면 코드를 수정할 때마다 서버가 자동 재시작된다
    app.run(debug=True)
