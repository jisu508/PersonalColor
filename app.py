from flask import Flask, render_template, request, jsonify
import os
import json # 추가된 부분

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
    
    file = request.files['image']
    
    # 📌 프론트엔드에서 보낸 가이드 박스 좌표 수신
    coords_json = request.form.get('coords')
    if coords_json:
        coords = json.loads(coords_json)
        print("전송받은 흰 종이 가이드 박스 좌표:", coords)
        # 추후 OpenCV로 이 비율(좌표) 영역만 크롭하여 화이트밸런스 기준점으로 사용

    # -------------------------------------------------------------
    # 추후 이 부분에 OpenCV와 MediaPipe 로직을 연결하면 됩니다.
    # 예: 
    # img_bytes = file.read()
    # image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    # corrected_img = apply_white_balance(image, coords) # 흰 종이 좌표 활용
    # face_data = extract_face_and_color(corrected_img) # MediaPipe 적용
    # -------------------------------------------------------------
    
    # 지금은 프론트엔드가 제대로 작동하는지 테스트하기 위해 임시 데이터를 반환합니다.
    # (새로 바뀐 UI 디자인 구조에 맞춰 리스트 형태로 변경했습니다)
    result_data = {
        "reliability": 92,
        "reliability_msg": "조명 충분",
        "percentages": [
            {"name": "겨울브라이트", "value": 78, "color": "#2B5299"},
            {"name": "여름라이트", "value": 12, "color": "#A8DCE7"},
            {"name": "가을뮤트", "value": 5, "color": "#B88354"},
            {"name": "봄브라이트", "value": 5, "color": "#F095B5"}
        ],
        "best_group": "겨울 쿨톤"
    }
    
    return jsonify(result_data)

if __name__ == '__main__':
    # 서버 실행 (debug=True로 설정하면 코드를 수정할 때마다 서버가 자동 재시작됩니다)
    app.run(debug=True)