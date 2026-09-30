"""
=====================================================================
 diagnose.py — 사진 넣고 진단 결과 바로 보기 (제일 자주 쓸 도구)
=====================================================================
 실행
   # 사진 한 장
   python tools/diagnose.py data/collected/webcam/사진.jpg

   # 폴더 전체 (한 줄씩 요약)
   python tools/diagnose.py data/collected/webcam

   # 판정 방법 바꿔서 비교
   python tools/diagnose.py 사진.jpg --method b
   python tools/diagnose.py 사진.jpg --compare      ← 5가지 방법을 한 번에 비교

   # 흰 종이 없이(연예인 사진 등)
   python tools/diagnose.py 사진.jpg --no-wb
=====================================================================
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.diagnosis.season import compare_methods, load_reference   # noqa: E402
from backend.image_io import imread                                    # noqa: E402
from backend.pipeline import run_pipeline                              # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def show_one(path: Path, args):
    r = run_pipeline(imread(path), method=args.method, skip_white_balance=args.no_wb)
    print("=" * 62)
    print(f"[{path.name}]")
    if not r["ok"]:
        print(f"  진단 실패: {r['message']}")
        return
    c = r["colors"]
    print(f"  피부 Lab   L {c['skin'][0]:5.1f}  a {c['skin'][1]:5.1f}  b {c['skin'][2]:5.1f}")
    if c["hair"]:
        print(f"  머리 Lab   L {c['hair'][0]:5.1f}  a {c['hair'][1]:5.1f}  b {c['hair'][2]:5.1f}")
    if c["eye"]:
        print(f"  눈   Lab   L {c['eye'][0]:5.1f}  a {c['eye'][1]:5.1f}  b {c['eye'][2]:5.1f}")
    d = r["season_detail"]
    print(f"\n  판정 방법 : {r['method']}   근거값 { {k: v for k, v in d.items() if k in ('D_bc','hue_deg','skin_b','D','threshold')} }")
    print(f"  웜/쿨     : {r['warm_cool']}   (웜 {d.get('warm_percent', '-')}%)")
    print(f"  결과      : {r['best_group']}")
    for p in r["percentages"]:
        bar = "█" * int(round(p["value"] / 3))
        print(f"     {p['name']:<7}{p['value']:5.1f}%  {bar}")
    print(f"  신뢰도    : {r['reliability']} ({r['reliability_msg']})")
    print(f"  해석      : {r['message']}")
    if r["warnings"]:
        print("  경고      :")
        for w in r["warnings"]:
            print(f"     - {w}")
    if args.compare:
        print("\n  [판정 방법 비교]")
        for m, v in compare_methods(c).items():
            if "error" in v:
                print(f"     {m:<8} 계산 불가 ({v['error'][:40]})")
            else:
                nums = {k: val for k, val in v.items() if k not in ("warm_cool", "score")}
                print(f"     {m:<8} {v['warm_cool']:<5} 점수 {v['score']:+6.2f}   {nums}")


def show_row(path: Path, args):
    r = run_pipeline(imread(path), method=args.method, skip_white_balance=args.no_wb)
    if not r["ok"]:
        print(f"  {path.name[:34]:<36} 실패: {r['message'][:34]}")
        return
    top = r["percentages"][0]
    print(f"  {path.name[:34]:<36} {r['warm_cool']:<5} 웜 {r['season_detail'].get('warm_percent', 0):5.1f}%   "
          f"{top['name']:<7}{top['value']:5.1f}%   신뢰도 {r['reliability']:3d}")


def main():
    ap = argparse.ArgumentParser(description="사진 진단 결과 보기")
    ap.add_argument("path", help="사진 파일 또는 폴더")
    ap.add_argument("--method", default=None, help="bc / hue / b / paper_d / smtc (기본: JSON 설정값)")
    ap.add_argument("--compare", action="store_true", help="5가지 판정 방법 비교")
    ap.add_argument("--no-wb", action="store_true", help="조명 보정 생략")
    args = ap.parse_args()

    p = Path(args.path)
    ref = load_reference()
    print(f"기본 판정 방법: {ref['warm_cool']['method']}  (기준값 {ref['warm_cool'][ref['warm_cool']['method']]['threshold']})")
    if p.is_dir():
        files = sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
        print(f"폴더 {p} — 사진 {len(files)}장\n")
        for f in files:
            show_row(f, args)
    else:
        show_one(p, args)


if __name__ == "__main__":
    main()
