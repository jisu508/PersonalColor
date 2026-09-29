"""
=====================================================================
 fit_threshold.py — 라벨이 붙은 사진으로 '웜/쿨 경계값' 찾기
=====================================================================
 하는 일
   1) 폴더 안 사진에서 피부·머리·눈 색을 뽑는다
   2) 판정 방법 3가지(b / paper_d / smtc)마다
      - 정답(웜/쿨)을 가장 잘 맞히는 경계값을 찾는다
      - 그때의 정확도를 계산한다
   3) 어떤 방법을 쓸지, 경계값을 얼마로 할지 알려준다
      → 마음에 들면 data/reference/season_reference.json 에 그 숫자를 적는다

 폴더 구조 (폴더 이름이 정답 라벨)
   data/collected/
       warm/   ← 웜으로 판정된 사람 사진
       cool/   ← 쿨로 판정된 사람 사진
   (celeb_warm, celeb_cool 처럼 앞에 뭐가 붙어도 warm/cool 이 들어 있으면 인식한다)

 실행
   # 연예인 사진처럼 흰 종이가 없는 경우 (조명 보정 없이)
   python tools/fit_threshold.py data/collected --no-wb

   # 흰 종이가 같이 찍힌 우리 촬영본
   python tools/fit_threshold.py data/anchor

   # 결과를 CSV로도 저장
   python tools/fit_threshold.py data/collected --no-wb --out outputs/fit.csv
=====================================================================
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.diagnosis.season import load_reference, warm_cool_score   # noqa: E402
from backend.image_io import imread                                    # noqa: E402
from backend.pipeline import run_pipeline                              # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
METHODS = ["bc", "hue", "b", "paper_d", "smtc"]
RAW = {"bc": lambda d: d.get("D_bc"), "hue": lambda d: d.get("hue_deg"), "b": lambda d: d.get("skin_b"), "paper_d": lambda d: d.get("D"),
       "smtc": lambda d: (d.get("cool_distance", 0) - d.get("warm_distance", 0))}


def label_of(path: Path, root: Path):
    """경로에 warm / cool 이 들어 있으면 그것을 정답으로 본다"""
    parts = [p.lower() for p in path.relative_to(root).parts]
    for p in parts:
        if "warm" in p or p in ("웜", "웜톤"):
            return "warm"
        if "cool" in p or p in ("쿨", "쿨톤"):
            return "cool"
    return None


def best_threshold(values, labels):
    """
    경계값 후보를 훑어 정확도가 가장 높은 값을 찾는다.
    (값이 클수록 웜이라고 가정)
    반환: (경계값, 정확도, 웜 정확도, 쿨 정확도)
    """
    v = np.asarray(values, float)
    y = np.array([1 if l == "warm" else 0 for l in labels])
    cands = np.unique(np.concatenate([v, (v[:-1] + v[1:]) / 2 if len(v) > 1 else v]))
    best = (None, -1, 0, 0)
    for t in cands:
        pred = (v > t).astype(int)
        acc = float((pred == y).mean())
        if acc > best[1]:
            wa = float((pred[y == 1] == 1).mean()) if (y == 1).any() else 0.0
            ca = float((pred[y == 0] == 0).mean()) if (y == 0).any() else 0.0
            best = (float(t), acc, wa, ca)
    return best


def main():
    ap = argparse.ArgumentParser(description="라벨 붙은 사진으로 웜/쿨 경계값 찾기")
    ap.add_argument("folder")
    ap.add_argument("--no-wb", action="store_true", help="조명 보정 생략 (흰 종이 없는 사진)")
    ap.add_argument("--out", default=None, help="사진별 측정값 CSV 저장 경로")
    args = ap.parse_args()

    root = Path(args.folder)
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"사진이 없습니다: {root}")

    ref = load_reference()
    rows = []
    print(f"사진 {len(files)}장 처리 (조명 보정 {'안 함' if args.no_wb else '함'})")
    for i, path in enumerate(files, 1):
        label = label_of(path, root)
        row = {"file": str(path.relative_to(root)), "label": label or ""}
        try:
            r = run_pipeline(imread(path), skip_white_balance=args.no_wb)
            if not r["ok"]:
                row["error"] = r["message"]
            else:
                colors = r["colors"]
                row.update({"skin_L": colors["skin"][0], "skin_a": colors["skin"][1], "skin_b": colors["skin"][2],
                            "hair_b": (colors["hair"] or [None, None, None])[2],
                            "eye_b": (colors["eye"] or [None, None, None])[2],
                            "top": r["top"], "reliability": r["reliability"]})
                for m in METHODS:
                    try:
                        _, detail = warm_cool_score(colors, ref, m)
                        row[m] = RAW[m](detail)
                    except Exception:
                        row[m] = None
        except Exception as e:
            row["error"] = f"{type(e).__name__}: {e}"
        rows.append(row)
        mark = row.get("error", "OK")
        print(f"  [{i}/{len(files)}] {row['file'][:45]:<45} {label or '라벨없음':<8} {str(mark)[:30]}")

    ok = [r for r in rows if r.get("label") in ("warm", "cool") and r.get("skin_b") is not None]
    print(f"\n분석 대상: {len(ok)}장 (웜 {sum(r['label']=='warm' for r in ok)} / 쿨 {sum(r['label']=='cool' for r in ok)})")
    if len(ok) < 4:
        print("라벨 붙은 사진이 너무 적어 경계값을 찾을 수 없습니다. (최소 4장, 권장 20장 이상)")
    else:
        print(f"\n{'방법':<10}{'경계값':>10}{'정확도':>9}{'웜 정확도':>11}{'쿨 정확도':>11}   설명")
        desc = {"bc": "b - C (현재 기본값)", "hue": "색상각 h", "b": "피부 b값만", "paper_d": "논문① 판별식 D",
                "smtc": "논문③·ShowMeTheColor"}
        for m in METHODS:
            vals = [r.get(m) for r in ok]
            if any(v is None for v in vals):
                print(f"{m:<10}{'계산 실패(부위 추출 실패 포함)':>40}")
                continue
            t, acc, wa, ca = best_threshold(vals, [r["label"] for r in ok])
            print(f"{m:<10}{t:10.2f}{acc:8.0%}{wa:11.0%}{ca:11.0%}   {desc[m]}")
        print("\n웜/쿨 그룹별 피부 b 평균")
        for lab in ("warm", "cool"):
            v = [r["skin_b"] for r in ok if r["label"] == lab]
            if v:
                print(f"  {lab:<5} n={len(v):3d}   b 평균 {np.mean(v):6.2f}   표준편차 {np.std(v):5.2f}   범위 {min(v):.1f} ~ {max(v):.1f}")
        print("\n※ 정확도가 60% 아래면 그 방법은 이 사진들에서 웜/쿨을 못 가르는 것입니다.")
        print("※ 쓸 방법을 정했으면 data/reference/season_reference.json 의 threshold 를 위 경계값으로 바꾸세요.")

    if args.out:
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        cols = ["file", "label", "skin_L", "skin_a", "skin_b", "hair_b", "eye_b", "b", "paper_d", "smtc",
                "top", "reliability", "error"]
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader(); w.writerows(rows)
        print(f"\nCSV 저장: {out}")


if __name__ == "__main__":
    main()
