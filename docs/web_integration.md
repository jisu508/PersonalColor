# 웹 연결 방법 (웹 담당 → 백엔드)

지금 `app.py` 의 `/api/diagnose` 는 가짜 데이터를 돌려주고 있습니다.
아래처럼 세 줄만 바꾸면 진짜 진단 결과가 나옵니다. **화면 코드(main.js, index.html)는 고칠 필요 없습니다.**

```python
from backend.image_io import imdecode_bytes      # 추가
from backend.pipeline import run_pipeline        # 추가

@app.route('/api/diagnose', methods=['POST'])
def diagnose():
    if 'image' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400

    img = imdecode_bytes(request.files['image'].read())

    coords = request.form.get('coords')
    guide_box = json.loads(coords) if coords else None   # 가이드 박스 좌표(비율 0~1 또는 픽셀)

    result = run_pipeline(img, guide_box=guide_box)

    if not result["ok"]:
        return jsonify({"error": result["message"]}), 400
    return jsonify(result)
```

## 돌아오는 값

화면에서 쓰는 값은 지금 쓰고 있는 이름 그대로입니다.

```json
{
  "ok": true,
  "percentages": [
    {"key": "winter_cool", "name": "겨울 쿨", "value": 50.0, "color": "#2B5299"},
    {"key": "summer_cool", "name": "여름 쿨", "value": 38.3, "color": "#A8DCE7"},
    {"key": "spring_warm", "name": "봄 웜",  "value": 7.4,  "color": "#F095B5"},
    {"key": "autumn_warm", "name": "가을 웜", "value": 4.3,  "color": "#B88354"}
  ],
  "best_group": "겨울 쿨",
  "reliability": 80,
  "reliability_msg": "참고용으로 충분",
  "message": "두 시즌 경계에 있어 양쪽 색을 모두 활용할 수 있습니다.",
  "warnings": ["흰 종이 영역이 균일하지 않습니다 ..."]
}
```

- `percentages` 는 **큰 값부터 정렬**되어 있고 합이 100입니다. 색상 코드까지 들어 있어 막대 그래프에 바로 씁니다.
- `warnings` 는 사용자에게 보여줄 안내문입니다. 있으면 결과 아래에 띄워주세요.
- 이 밖에 `colors`, `quality`, `lighting`, `season_detail` 같은 값도 함께 오는데, 화면에는 안 써도 됩니다(디버그용).

## 실패할 때

`ok: false` 와 함께 아래 중 하나가 옵니다. `message` 를 그대로 보여주면 됩니다.

| error | 뜻 |
|---|---|
| `paper_not_found` | 흰 종이를 못 찾음 |
| `face_not_found` | 얼굴을 못 찾음 |
| `white_balance_failed` | 가이드 박스 좌표가 사진 밖 |

## 가이드 박스 좌표 형식

둘 다 됩니다.
- 비율: `{"x1":0.6,"y1":0.3,"x2":0.85,"y2":0.7}` (0~1) — 화면 크기와 사진 크기가 달라도 알아서 맞춥니다
- 픽셀: `{"x1":1190,"y1":420,"x2":1450,"y2":680}`

좌표를 안 보내면 백엔드가 사진에서 흰 종이를 **자동으로 찾습니다**. 다만 배경이 밝으면 잘못 찾을 수 있으니, 가이드 박스 좌표를 보내주는 쪽이 정확합니다.
