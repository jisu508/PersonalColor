"""
=====================================================================
 batch_extract.py — 여러 장의 사진에서 부위별 색을 한 번에 뽑아 CSV 로 저장
=====================================================================
 누가 쓰나 : 알고리즘·문서 파트 (수집한 인물 사진으로 시즌 기준표를 만들 때)

 하는 일
   폴더 안 모든 사진 → 조명 보정 → 얼굴 자동 검출 → 피부·눈·머리 Lab + 사진 품질
   → CSV 한 줄씩 저장. 엑셀/Jupyter 에서 바로 시즌별 평균·분포를 볼 수 있다.

 정답 시즌(라벨) 붙이는 법 — 둘 중 아무거나
   ① 시즌별 폴더 :  data/collected/spring_warm/아무이름.jpg
   ② 파일명 앞머리:  data/collected/spring_warm_아무이름.jpg

 조명 보정 방식 (--normalize)
   grayworld : 기본값. 흰 종이가 없는 수집 사진용 **대체** 보정 (부정확함, lighting.py 설명 참고)
   none      : 보정 없이 원본 그대로
   ※ 흰 종이가 찍힌 우리 촬영 사진은 이 도구 말고 check_face_color.py 를 쓸 것

 실행:
   python tools/batch_extract.py data/collected --out outputs/collected.csv
   python tools/batch_extract.py data/collected --normalize none

 ★ 중요 : 이렇게 뽑은 수치는 '시즌 사이의 상대적 경향'을 보는 데 쓴다.
   수집 사진은 조명·카메라·보정이 제각각이라 절대값(특히 b 값)을 그대로
   기준표에 쓰면 시연 환경과 어긋난다. (docs/reference_table_plan.md 참고)
=====================================================================
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.image_io import imread                                     # noqa: E402
from backend.vision.face_color import extract_face_colors               # noqa: E402
from backend.vision.landmarks import FaceNotFoundError                  # noqa: E402
from backend.vision.lighting import apply_gains, gray_world_gains       # noqa: E402

SEASONS = ["spring_warm", "summer_cool", "autumn_warm", "winter_cool"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def guess_season(path: Path, root: Path) -> str:
    """폴더 이름 또는 파일명 앞머리에서 시즌 라벨을 읽는다. 못 찾으면 빈 값."""
    for part in path.relative_to(root).parts:
        for s in SEASONS:
            if part == s or part.lower().startswith(s):
                return s
    return ""


def main():
    ap = argparse.ArgumentParser(description="사진 여러 장 → 부위별 Lab CSV")
    ap.add_argument("folder", help="사진이 들어있는 폴더 (하위 폴더도 같이 훑음)")
    ap.add_argument("--out", default=None, help="CSV 저장 경로 (기본: outputs/<폴더이름>.csv)")
    ap.add_argument("--normalize", choices=["grayworld", "none"], default="grayworld")
    args = ap.parse_args()

    root = Path(args.folder)
    if not root.is_dir():
        raise SystemExit(f"폴더가 없습니다: {root}")
    out_path = Path(args.out) if args.out else ROOT / "outputs" / f"{root.name}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    print(f"사진 {len(files)}장 처리 시작 (보정: {args.normalize})")

    cols = (["file", "season", "ok", "error"]
            + [f"{part}_{c}" for part in ("skin", "eye", "hair") for c in "Lab"]
            + ["brightness", "sharpness", "face_width_ratio", "asymmetry", "num_faces", "warnings"])

    n_ok = 0
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글 안 깨짐
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i, path in enumerate(files, 1):
            row = {"file": str(path.relative_to(root)), "season": guess_season(path, root), "ok": 0, "error": ""}
            try:
                img = imread(path)
                if args.normalize == "grayworld":
                    img = apply_gains(img, gray_world_gains(img))
                fc = extract_face_colors(img)
                d = fc.to_dict()
                for part in ("skin", "eye", "hair"):
                    lab = d[part]
                    for j, c in enumerate("Lab"):
                        row[f"{part}_{c}"] = "" if lab is None else lab[j]
                row.update(d["quality"])
                row["num_faces"] = d["face"]["num_faces"]
                row["warnings"] = " / ".join(d["warnings"])
                row["ok"] = 1 if d["skin"] else 0
                n_ok += row["ok"]
            except FaceNotFoundError as e:
                row["error"] = str(e)
            except Exception as e:                    # 사진 한 장이 깨져도 전체가 멈추지 않게
                row["error"] = f"{type(e).__name__}: {e}"
            w.writerow(row)
            print(f"  [{i}/{len(files)}] {row['file']:<40} {'OK' if row['ok'] else 'FAIL ' + row['error'][:40]}")

    print(f"\n성공 {n_ok} / {len(files)}장  →  {out_path}")
    print("엑셀에서 시즌별 평균을 내보고, 시즌 사이에 차이가 나는 값(특히 b, a)이 무엇인지 확인하세요.")


if __name__ == "__main__":
    main()
