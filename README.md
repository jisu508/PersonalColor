# 근거를 보여주는 AI 퍼스널컬러 진단

수원대 정보보호학과 · 시스템보안프로젝트 2026-2학기 (4인 팀)

사진/웹캠으로 퍼스널컬러(봄웜·여름쿨·가을웜·겨울쿨)를 **퍼센티지**로 진단하는 로컬 웹 서비스.

| 차별점 | 내용 |
|---|---|
| ① 조명 보정 | 사진 속 흰 종이를 흰색(255) 기준으로 화이트밸런스 → 조명이 달라도 일관된 색 |
| ② 퍼센티지 | "겨울쿨"이라 단정하지 않고 시즌별 % 로 표시 |
| ③ 신뢰도 | 사진 품질 + 결과 명확도로 0~100% 신뢰도 표시 |

외부 API·DB 없음. 모든 처리는 로컬 Python (OpenCV · MediaPipe · NumPy · Flask).

---

## 1. 폴더 구조와 담당

```
personal_color/
├── README.md                  ← 지금 이 문서                              [문서]
├── requirements.txt           ← 설치할 라이브러리 목록
├── .gitignore                 ← 사진·결과물은 GitHub에 안 올라가게
│
├── backend/                   ← 진단 로직 전부 (웹과 무관한 순수 Python)
│   ├── color_utils.py         공용: BGR→Lab 변환, 영역 평균색, ΔE 거리    [공용]
│   ├── image_io.py            공용: 한글 경로 안전한 사진 읽기/저장        [공용]
│   ├── vision/                ── 사진 → 색 숫자 ──                        [영상처리]
│   │   ├── lighting.py        ✅ 2단계  흰 종이 기준 조명 보정
│   │   ├── landmarks.py       ✅ 3단계  MediaPipe 얼굴 478점(홍채 포함) 검출
│   │   └── face_color.py      ✅ 3단계  볼·이마·홍채 → 피부·눈 Lab (머리는 2주차)
│   ├── diagnosis/             ── 색 숫자 → 판정 ──                        [알고리즘]
│   │   ├── season.py          ⬜ 4단계  기준표 거리 → softmax → 퍼센티지
│   │   └── confidence.py      ⬜ 5단계  신뢰도 0~100%
│   └── pipeline.py            ⬜ 6단계  사진 → 결과 dict (위 모듈 연결)
│
├── web/                       ⬜ 7단계  Flask + getUserMedia 촬영 화면    [웹]
│   ├── templates/  static/js/  static/css/
│
├── models/                    MediaPipe 모델 파일 (face_landmarker.task, 레포에 포함)
│
├── data/
│   ├── reference/             시즌 기준표 JSON (4단계)                    [알고리즘]
│   └── samples/               테스트 사진 (GitHub 업로드 금지)
│
├── tools/                     사람이 직접 돌려보는 확인용 스크립트
│   ├── find_coords.py         사진 클릭 → 좌표 박스 출력 (구 find_coords.py)
│   ├── check_lighting.py      조명 보정 전/후 비교 (구 check2.py)
│   ├── draw_landmarks.py      얼굴 478점이 제대로 잡히는지 그려보기
│   └── check_face_color.py    부위별 색 자동 추출 + 두 사진 비교
│
├── tests/                     자동 테스트 (pytest)                        [테스트]
│   ├── test_lighting.py
│   └── test_face_color.py
├── notebooks/                 Jupyter 실험 (기준표 튜닝 등)
├── docs/                      일정표·데이터 형식(data_format.md)·실험 기록   [문서]
└── outputs/                   실행 결과 이미지 (자동 생성, 커밋 X)
```

### 모듈 규칙 (파트끼리 안 부딪히기 위한 약속)

1. **`vision/` ↔ `diagnosis/` 는 서로 import 하지 않는다.** 둘을 잇는 곳은 `pipeline.py` 하나.
   → 알고리즘 파트는 사진 없이 Lab 숫자만으로 테스트·튜닝할 수 있다.
2. **`web/` 은 계산하지 않는다.** `backend.pipeline.run_pipeline()` 만 호출.
3. **숫자(기준값·임계값)는 코드에 흩뿌리지 않는다.** 기준표는 `data/reference/*.json`,
   품질 임계값은 각 모듈 상단 `THRESHOLDS` 에 모아둔다.
4. 사진 읽기/저장은 `cv2.imread` 대신 `backend.image_io.imread / imwrite` (Windows 한글 경로 문제).

### 데이터 흐름

```
[웹캠 촬영] → lighting.white_balance → face_color.extract_face_colors
            → season.diagnose_season → confidence.compute_confidence → [결과 화면]
```

---

## 2. 설치 (Windows PowerShell 기준, 처음 한 번만)

Python 3.9 이상 필요 (`python --version` 으로 확인).

```powershell
# (1) 코드 받기
git clone https://github.com/jisu508/PersonalColor.git
cd PersonalColor

# (2) 라이브러리 설치
python -m pip install -r requirements.txt

# (3) 설치 확인 — 사진이 없으면 "16 passed, 3 skipped", 사진까지 넣었으면 "19 passed" 가 정상
#     (MediaPipe 가 W0000 ... 같은 로그를 출력하는데 오류가 아니므로 무시)
python -m pytest tests -v
```

<details>
<summary>가상환경(.venv)을 쓰고 싶다면 (선택)</summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # 맥/리눅스: source .venv/bin/activate
python -m pip install -r requirements.txt
```
"스크립트를 실행할 수 없습니다" 오류가 나면 한 번만 실행 후 다시 활성화:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
</details>

### 테스트 사진 넣기 (필수)

인물 사진은 개인정보라 **GitHub에 올리지 않습니다** (`.gitignore` 처리).
팀 드라이브/카톡에서 `photo1.jpg`, `photo3.jpg` 를 받아 **`data/samples/` 폴더에 직접 넣으세요.**

```powershell
# 예) 사진이 바탕화면\시보프\1차발표 에 있을 때
Copy-Item "$HOME\Desktop\시보프\1차발표\photo1.jpg", "$HOME\Desktop\시보프\1차발표\photo3.jpg" data\samples\
dir data\samples          # photo1.jpg, photo3.jpg 가 보이면 OK
```

사진이 없으면 `FileNotFoundError: 사진 파일이 없습니다` 오류가 납니다.

## 3. 실행 방법 (현재 2단계까지)

모든 명령은 **프로젝트 최상위 폴더**(README.md 가 있는 폴더)에서 실행합니다.
긴 명령은 **한 줄로** 붙여넣으세요. (여러 줄로 나눌 땐 PowerShell은 줄 끝에 백틱 `` ` ``, 맥/리눅스는 `\`)

### (1) 자동 테스트 — 코드 고친 뒤엔 항상

```powershell
python -m pytest tests -v
```

### (2) 좌표 찾기 — 새 사진을 쓸 때만

```powershell
python tools/find_coords.py data/samples/photo1.jpg
```

1. 사진 창("click")이 뜸 (안 보이면 작업표시줄 확인)
2. **흰 종이 가운데** 클릭 → PowerShell에 `박스 645,1817,725,1897` 처럼 출력
3. **볼 가운데** 클릭 → 같은 방식으로 출력
4. 사진 창을 선택한 상태에서 **아무 키**나 누르면 종료
5. 출력된 박스 값을 (3)의 `--white`(종이), `--obj`(볼) 에 붙여넣기

### (3) 조명 보정 확인 — photo1(노란 조명) vs photo3(밝은 조명)

```powershell
python tools/check_lighting.py --photo data/samples/photo1.jpg --white 645,1817,725,1897 --obj 1116,1827,1196,1907 --photo data/samples/photo3.jpg --white 745,3130,825,3210 --obj 2144,3111,2224,3191
```

정상 출력 (마지막 부분):
```
두 사진의 물건색 차이(ΔE) | 보정 전:  22.8   보정 후:  15.2
성공: 보정 후 두 사진의 색 차이가 줄었습니다.
```

결과 이미지 열기 (위=원본, 아래=보정 후 / 빨강=종이, 초록=볼):
```powershell
start outputs\lighting_compare.jpg
```

사진 1장만 확인: `python tools/check_lighting.py --photo data/samples/photo1.jpg --white 645,1817,725,1897`

### (4) 얼굴 점(랜드마크) 확인 — 3단계

```powershell
python tools/draw_landmarks.py data/samples/photo1.jpg
```
→ `outputs\landmarks_photo1.jpg` (초록=얼굴 점, 빨강=홍채 점). `--numbers` 를 붙이면 점 번호까지 표시.

### (5) 얼굴 부위별 색 자동 추출 — 3단계 (좌표 클릭 필요 없음)

```powershell
python tools/check_face_color.py --photo data/samples/photo1.jpg --white 645,1817,725,1897 --photo data/samples/photo3.jpg --white 745,3130,825,3210
```

- 사진마다 볼·이마·홍채의 Lab, 사용 여부(O/X), 제외 이유 출력
- 마지막에 두 사진의 **피부·눈 색 차이(보정 전 → 후)** 비교
- 영역 그림: `outputs\face_color_photo1.jpg` (초록=사용, 빨강=제외, 하늘색=눈꺼풀 윤곽)
- `--white` 를 빼면 조명 보정 없이 추출만 확인


코드에서 쓰기:

```python
from backend.image_io import imread
from backend.color_utils import mean_lab
from backend.vision.lighting import white_balance

img = imread("data/samples/photo1.jpg")
wb = white_balance(img, white_box=(645, 1817, 725, 1897))
print(mean_lab(wb.image, box=(1116, 1827, 1196, 1907)))   # 보정 후 볼 Lab
print(wb.warnings)                                         # 품질 경고 (없으면 [])

from backend.vision.face_color import extract_face_colors
fc = extract_face_colors(wb.image)                         # 얼굴 자동 검출 → 부위별 색
print(fc.skin, fc.eye)                                     # 피부 Lab, 눈 Lab
print(fc.to_dict())                                        # 알고리즘 파트로 넘기는 형식 (docs/data_format.md)
```

## 4. 진행 현황

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | 폴더 구조 · README | ✅ |
| 2 | 조명 보정 함수화 (check2.py 기반) | ✅ photo1·photo3 실측 22.8 → 15.2 (check2 원본 결과와 동일) |
| 3 | MediaPipe 얼굴·색 자동 추출 | 🟡 피부·눈 ✅ (피부 a·b 차이 15.2 → 3.4) / 머리카락·품질 지표는 2주차 |
| 4 | 시즌 퍼센티지 판정 | ⬜ |
| 5 | 신뢰도 | ⬜ |
| 6 | 파이프라인 연결 | ⬜ |
| 7 | 웹 (Flask + 웹캠) | ⬜ |

## 5. Git 협업 (main 에 바로 push 하는 방식)

레포: https://github.com/jisu508/PersonalColor

### 매번 작업할 때 순서

```powershell
git pull origin main               # ① 작업 시작 전: 팀원이 올린 최신 코드 받기

# ... 코드 작업 ...

python -m pytest tests             # ② 올리기 전: 테스트 통과 확인
git add .
git status                         #    .jpg 사진 / .venv / __pycache__ 가 목록에 없는지 확인!
git commit -m "무엇을 했는지 한 줄로"
git pull origin main               # ③ 올리기 직전: 그사이 올라온 코드 한 번 더 받기
git push origin main               # ④ 올리기
```

### 팀 규칙

1. **작업 전과 push 직전에 항상 `git pull origin main`.**
   push 가 `! [rejected] ... (fetch first)` 로 거절되면 → 그사이 누가 먼저 올린 것.
   `git pull origin main` 후 다시 `git push origin main` 하면 된다.
2. **push 전 `python -m pytest tests` 통과 확인.** main 에 바로 반영되므로 고장 난 코드를 올리면 팀 전원이 멈춘다.
3. **자기 파트 폴더 위주로 수정.** 같은 파일을 동시에 고치면 충돌이 난다.
   README·requirements.txt 같은 공통 파일을 고칠 땐 단톡에 먼저 알리기.
4. **인물 사진은 절대 커밋하지 않기** (`data/samples/`, `outputs/` 는 `.gitignore` 처리됨).

### pull 할 때 충돌(CONFLICT)이 나면

```
CONFLICT (content): Merge conflict in 파일이름
```
1. 해당 파일을 열면 `<<<<<<<`, `=======`, `>>>>>>>` 표시가 있다. 위쪽이 내 코드, 아래쪽이 팀원 코드.
2. 둘 중 맞는 내용만 남기고 표시 세 줄을 지운다. (모르겠으면 그 파일 담당 팀원에게 물어보기)
3. `git add 파일이름` → `git commit -m "충돌 해결"` → `git push origin main`
