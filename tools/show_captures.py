r"""
웹에서 촬영한 기록(data/captures/)을 표로 보여준다.

  python tools\show_captures.py           최근 20건
  python tools\show_captures.py --all      전부

각 줄의 뜻
  h      색상각(도). 이 숫자 하나로 웜/쿨이 갈린다 (기준선은 season_reference.json)
  a, b   피부 Lab 의 a(붉은기), b(노란기)
  L, C   피부 밝기, 채도 — 봄/가을, 여름/겨울을 가르는 축
  gain   화이트밸런스 보정 배율 (B, G, R). R 이 크면 원본이 파랬고, B 가 크면 원본이 누랬다는 뜻
  종이    흰 종이 기준색 (보정 전 BGR). 세 숫자가 비슷할수록 진짜 흰색
  방식    auto = 코드가 종이를 스스로 찾음 / guide = 웹 가이드 박스 좌표를 씀
  종이위치 잡은 종이 영역 [x1,y1,x2,y2]. 찍을 때마다 확 달라지면 엉뚱한 데를 잡는 것
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CAPTURES = Path("data/captures")


def row(d: Path):
    try:
        r = json.loads((d / "result.json").read_text(encoding="utf-8"))
    except Exception:
        return None
    det = r.get("season_detail") or {}
    skin = (r.get("colors") or {}).get("skin") or [None] * 3
    light = r.get("lighting") or (r.get("debug") or {}).get("lighting") or {}
    gains = light.get("gains_bgr") or [None] * 3
    paper = light.get("paper_bgr") or [None] * 3
    tone = r.get("tone") or (r.get("debug") or {}).get("tone") or {}
    pinfo = light.get("paper") or {}
    mode = pinfo.get("mode", "-")
    pbox = pinfo.get("box")
    pbox_s = f"{pbox[0]},{pbox[1]}-{pbox[2]},{pbox[3]}" if pbox else "-"
    f = lambda v, n=1: "-" if v is None else f"{v:.{n}f}"
    return (d.name, f(det.get("hue_deg")), f(skin[1]), f(skin[2]), f(skin[0]), f(det.get("skin_C")),
            f"{f(gains[0],2)}/{f(gains[1],2)}/{f(gains[2],2)}",
            f"{f(paper[0],0)},{f(paper[1],0)},{f(paper[2],0)}",
            mode, pbox_s,
            f"{tone.get('label','-')} {f(tone.get('warm'))}%",
            r.get("best_group") or (r.get("error") or "")[:16])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    dirs = sorted([p for p in CAPTURES.glob("*") if p.is_dir()])
    if not dirs:
        print(f"{CAPTURES} 에 기록이 없습니다. 웹에서 촬영하면 한 건씩 쌓입니다.")
        return
    if not a.all:
        dirs = dirs[-20:]
    head = ("시각", "h", "a", "b", "L", "C", "gain B/G/R", "종이 BGR", "방식", "종이위치", "판정", "결과")
    w = (18, 6, 6, 6, 6, 6, 18, 16, 7, 20, 13, 18)
    print("".join(f"{h:<{x}}" for h, x in zip(head, w)))
    print("-" * sum(w))
    for d in dirs:
        r = row(d)
        if r:
            print("".join(f"{str(c):<{x}}" for c, x in zip(r, w)))
    print("\n  방식이 guide 면 웹 가이드 박스를, auto 면 코드가 찾은 종이를 기준으로 보정한 것입니다.")
    print("  같은 자리에서 찍었는데 '종이위치'가 확 달라지면 종이가 아닌 곳을 잡고 있다는 뜻입니다.")
    print(f"\n총 {len(dirs)}건. 사진은 {CAPTURES}\\<시각>\\original.jpg / corrected.jpg")


if __name__ == "__main__":
    main()
