"""
Flask 서버 — 메인 화면 + 촬영·진단 화면 + 진단 API

  GET  /               메인 화면 (templates/index.html)
  GET  /diagnose       촬영·진단 화면 (templates/diagnose.html)
  POST /api/diagnose   사진(+흰 종이 가이드 박스 좌표) → 진단 결과 JSON

진단 계산은 전부 backend/ 안에 있고, 여기서는 run_pipeline() 하나만 부른다.
결과 형식은 docs/web_integration.md 참고.

실행:  python app.py   →  http://127.0.0.1:5000
"""
import base64
import io
import json
import os
import re
import socket
import traceback
from datetime import datetime
from pathlib import Path

import cv2
from flask import (Flask, abort, jsonify, render_template, request,
                   send_from_directory, Response)

from backend.image_io import imdecode_bytes, imread, imwrite
from backend.pipeline import run_pipeline
from backend.report import (build_report, make_card, make_page_colors,
                            make_page_style, make_long)

app = Flask(__name__)

# 촬영 기록 남기기 — 왜 그런 판정이 나왔는지 나중에 확인하려면 숫자가 남아 있어야 한다.
# 끄고 싶으면 실행 전에  set SAVE_CAPTURES=0
SAVE_CAPTURES = os.environ.get("SAVE_CAPTURES", "1") != "0"
CAPTURE_DIR = Path("data/captures")


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


def _log_error(e: BaseException) -> None:
    """오류를 data/captures/errors.log 에 쌓는다. 다른 컴퓨터에서 난 오류를 확인할 때 쓴다."""
    try:
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        with (CAPTURE_DIR / "errors.log").open("a", encoding="utf-8") as f:
            f.write(f"\n===== {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
            f.write(f"{type(e).__name__}: {e}\n")
            f.write(traceback.format_exc())
    except Exception:
        pass


def _save_capture(original, corrected, result: dict) -> str | None:
    """촬영 원본·보정본·판정 숫자를 data/captures/ 에 남기고, 그 폴더 이름(ID)을 돌려준다.

    이 ID 가 리포트 주소가 된다 →  /report/<ID>
    """
    try:
        stamp = datetime.now().strftime("%m%d_%H%M%S_%f")[:-3]   # 밀리초까지 (같은 초에 두 장 찍어도 안 겹치게)
        d = CAPTURE_DIR / stamp
        d.mkdir(parents=True, exist_ok=True)
        imwrite(d / "original.jpg", original)          # 한글 경로 안전 저장
        if corrected is not None:
            imwrite(d / "corrected.jpg", corrected)
        keep = {k: v for k, v in result.items() if k != "corrected_image"}
        (d / "result.json").write_text(json.dumps(keep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[촬영 기록] {d}")
        return stamp
    except Exception:                      # 기록 실패가 진단을 막으면 안 된다
        traceback.print_exc()
    return None


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


# 1-0. 촬영·진단 화면 — 메인의 '무료 진단' / '지금 진단하기' 버튼이 여기로 넘어온다.
#      (함수 이름을 diagnose_page 로 둔 이유: 아래 API 함수 diagnose 와 이름이 겹치면 Flask 가 오류를 낸다)
@app.route('/diagnose')
def diagnose_page():
    return render_template('diagnose.html')


# 1-1. 브라우저가 자동으로 찾는 아이콘 — 없으면 콘솔에 빨간 404 가 뜨므로 빈 응답을 준다
@app.route('/favicon.ico')
def favicon():
    return '', 204


# 2-0. 어떤 오류가 나도 HTML 대신 JSON 으로 돌려준다.
#      (기본 Flask 는 오류 때 HTML 페이지를 주는데, 프론트가 JSON.parse 하다 터진다)
@app.errorhandler(Exception)
def handle_any_error(e):
    traceback.print_exc()                     # 터미널에는 전체 오류 내용을 그대로 남긴다
    _log_error(e)                             # data/captures/errors.log 에도 남긴다
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
    try:
        result = run_pipeline(img, guide_box=guide_box, return_images=True)
    except Exception as e:
        # 예상 못한 오류는 500 으로 터뜨리지 않고 '다시 촬영' 안내로 바꾼다.
        # (원인은 터미널과 data/captures/errors.log 에 남는다)
        traceback.print_exc()
        _log_error(e)
        if SAVE_CAPTURES:
            _save_capture(img, None, {"ok": False, "error": f"{type(e).__name__}: {e}"})
        return jsonify({
            "ok": False, "code": "pipeline_failed",
            "error": f"이 사진은 분석하지 못했습니다 ({type(e).__name__}). "
                     f"얼굴과 흰 종이가 함께 보이게 다시 촬영해주세요.",
        }), 400

    # 보정된 사진을 화면에 보여줄 수 있게 data URL 로 바꿔 넣는다 ('근거를 보여주는' 부분)
    images = result.pop("images", None) or {}
    result = _jsonable(result)          # numpy 숫자를 파이썬 숫자로 (jsonify 가 numpy 를 못 다룬다)

    capture_id = _save_capture(img, images.get("corrected"), result) if SAVE_CAPTURES else None
    result["capture_id"] = capture_id
    # 폰으로 열 주소라 localhost 가 아니라 공유기가 준 주소여야 한다
    result["report_url"] = f"http://{_lan_ip()}:5000/report/{capture_id}" if capture_id else None
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


# ─────────────────────────────────────────────────────────────
#  결과 요약 리포트 — 진단이 끝나면 QR 로 폰에서 열어본다
# ─────────────────────────────────────────────────────────────
CAPTURE_ID_RE = re.compile(r"^\d{4}_\d{6}_\d{1,6}$")      # 예) 1111_093706_522
DRAPE_FILES = {"drape_best.jpg", "drape_worst.jpg", "corrected.jpg", "full.png"}

# 시즌별 가이드 사진 (static/images/)
GUIDE_IMAGES = {
    "spring_warm": ["최종_봄.png", "봄_화장품.png", "봄_코디.png"],
    "summer_cool": ["최종_여름.png", "여름_화장품.png", "여름_코디.png"],
    "autumn_warm": ["최종_가을.png", "가을_화장품.png", "가을_코디.png"],
    "winter_cool": ["최종_겨울.png", "겨울_화장품.png", "겨울_코디.png"],
}


def _lan_ip() -> str:
    """공유기가 이 노트북에 준 주소. 폰에서 QR 로 접속하려면 이 주소여야 한다.

    127.0.0.1 은 노트북 자기 자신만 가리켜서 폰에서는 안 열린다.
    (실제로 패킷을 보내지는 않고, 라우팅 테이블만 확인한다)
    """
    try:
        sk = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sk.connect(("8.8.8.8", 80))
        ip = sk.getsockname()[0]
        sk.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _capture_dir(capture_id: str) -> Path:
    """촬영 기록 폴더. 주소에 들어온 값이라 형식을 반드시 검사한다(경로 탈출 방지)."""
    if not CAPTURE_ID_RE.match(capture_id or ""):
        abort(404)
    d = CAPTURE_DIR / capture_id
    if not (d / "result.json").exists():
        abort(404)
    return d


@app.route("/report/<capture_id>")
def report_page(capture_id):
    d = _capture_dir(capture_id)
    result = json.loads((d / "result.json").read_text(encoding="utf-8"))

    # 비교 사진은 '조명 보정본' 을 쓴다. 원본은 방 조명 색이 섞여 있어
    # 어떤 색이 받는지 판단하는 데 쓰면 안 된다. 없을 때만 원본으로 넘어간다.
    corrected, used = None, None
    for name in ("corrected.jpg", "original.jpg"):
        if (d / name).exists():
            try:
                corrected = imread(d / name)
                used = name
                break
            except Exception:
                pass

    r = build_report(result, capture_dir=d, corrected_bgr=corrected, photo_source=used)

    base = Path(app.static_folder) / "images"
    guide = [f"/static/images/{n}" for n in GUIDE_IMAGES.get(r["season_key"], [])
             if (base / n).exists()]

    return render_template("report.html", r=r, capture_id=capture_id, guide_images=guide)


@app.route("/report/<capture_id>/img/<filename>")
def report_image(capture_id, filename):
    """리포트에 들어가는 사진. data/captures/ 는 static 밖이라 따로 내보낸다."""
    d = _capture_dir(capture_id)
    if filename not in DRAPE_FILES:
        abort(404)
    return send_from_directory(d.resolve(), filename)


PAGE_MAKERS = {"1": None, "2": make_page_colors, "3": make_page_style}


def _report_of(capture_id):
    """리포트 dict 와 촬영 폴더를 함께 돌려준다 (여러 라우트가 같이 쓴다)"""
    d = _capture_dir(capture_id)
    result = json.loads((d / "result.json").read_text(encoding="utf-8"))
    corrected, used = None, None
    for name in ("corrected.jpg", "original.jpg"):
        if (d / name).exists():
            try:
                corrected = imread(d / name)
                used = name
                break
            except Exception:
                pass
    return d, build_report(result, capture_dir=d, corrected_bgr=corrected, photo_source=used)


@app.route("/report/<capture_id>/full.png")
def report_full_png(capture_id):
    """리포트 전체를 긴 사진 한 장으로. PIL 로 그리므로 외부 브라우저가 필요 없다.

    한 번 만들면 data/captures/<id>/full.png 에 저장해두고 다시 쓴다.
    """
    d = _capture_dir(capture_id)
    out = d / "full.png"
    if not out.exists():
        r = _report_of(capture_id)[1]
        out.write_bytes(make_long(r, d, Path(app.static_folder) / "images"))
    headers = {"Cache-Control": "no-store"}
    if request.args.get("dl"):
        headers["Content-Disposition"] = \
            f'attachment; filename="personal-color-{capture_id}.png"'
    return Response(out.read_bytes(), mimetype="image/png", headers=headers)


@app.route("/report/<capture_id>/page<int:n>.png")
def report_page_img(capture_id, n):
    """리포트 전체를 이미지 여러 장으로. 1=요약·비교 2=색 3=스타일"""
    if str(n) not in PAGE_MAKERS:
        abort(404)
    maker = PAGE_MAKERS[str(n)]
    d, r = _report_of(capture_id)
    png = make_card(r, d) if maker is None else maker(r)

    # 기본은 '인라인'. attachment 로 내려보내면 브라우저가 파일로만 취급해서
    # 폰에서 이미지를 길게 눌러도 "사진에 저장" 이 안 뜬다.
    # 눌러서 받는 버튼만 ?dl=1 을 붙여 attachment 로 준다.
    headers = {"Cache-Control": "no-store"}
    if request.args.get("dl"):
        headers["Content-Disposition"] = \
            f'attachment; filename="personal-color-{capture_id}-{n}.png"'
    return Response(png, mimetype="image/png", headers=headers)


@app.route("/report/<capture_id>/card.png")
def report_card(capture_id):
    """저장용 카드 이미지. QR 로 들어온 사람이 눌러서 갤러리에 담아간다."""
    d = _capture_dir(capture_id)
    result = json.loads((d / "result.json").read_text(encoding="utf-8"))

    corrected, used = None, None
    for name in ("corrected.jpg", "original.jpg"):
        if (d / name).exists():
            try:
                corrected = imread(d / name)
                used = name
                break
            except Exception:
                pass

    r = build_report(result, capture_dir=d, corrected_bgr=corrected, photo_source=used)
    png = make_card(r, d)
    return Response(png, mimetype="image/png", headers={
        "Content-Disposition": f'attachment; filename="personal-color-{capture_id}.png"',
        "Cache-Control": "no-store",
    })


@app.route("/qr/<capture_id>.png")
def qr_png(capture_id):
    """리포트 주소를 담은 QR. 주소는 서버가 직접 만든다(외부 입력 안 받음)."""
    _capture_dir(capture_id)
    try:
        import qrcode
    except ImportError:
        # 라이브러리가 없으면 QR 대신 안내 (화면에서는 주소를 글자로 보여준다)
        abort(503)
    url = f"http://{_lan_ip()}:5000/report/{capture_id}"
    img = qrcode.make(url, box_size=8, border=2)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), mimetype="image/png",
                    headers={"Cache-Control": "no-store"})


if __name__ == '__main__':
    # host='0.0.0.0' 이어야 같은 와이파이의 폰에서 QR 로 접속할 수 있다
    # (기본값은 노트북 자기 자신만 접속 가능)
    print(f"\n  이 컴퓨터:  http://127.0.0.1:5000")
    print(f"  폰에서:     http://{_lan_ip()}:5000   ← 같은 와이파이여야 합니다\n")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)