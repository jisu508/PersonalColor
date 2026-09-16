from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

# 1. 메인 화면(HTML) 띄우기
@app.route('/')
def home():
    return render_template('index.html')

# 2. 사진 업로드 및 진단 API
@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    # 프론트엔드에서 보낸 이미지 파일 받기
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400
    
    file = request.files['image']
    
    # -------------------------------------------------------------
    # 추후 이 부분에 OpenCV와 MediaPipe 로직을 연결하면 됩니다.
    # 예: 
    # img_bytes = file.read()
    # image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), -1)
    # corrected_img = apply_white_balance(image) # 흰 종이 기준 조명 보정
    # face_data = extract_face_and_color(corrected_img) # MediaPipe 적용
    # -------------------------------------------------------------
    
    # 지금은 프론트엔드가 제대로 작동하는지 테스트하기 위해 임시 데이터를 반환합니다.
    result_data = {
        "reliability": 92,
        "reliability_msg": "신뢰도 92%, 조명 충분",
        "percentages": {
            "primary": {"name": "겨울 브라이트", "value": 78},
            "secondary": {"name": "여름 라이트", "value": 12}
        },
        "analysis_msg": "전형적인 겨울 브라이트 타입입니다. 명도가 높은 원색 계열이 잘 어울립니다."
    }
    
    return jsonify(result_data)

if __name__ == '__main__':
    # 서버 실행 (debug=True로 설정하면 코드를 수정할 때마다 서버가 자동 재시작됩니다)
    app.run(debug=True)