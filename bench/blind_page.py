#!/usr/bin/env python3
"""
Страница слепого прослушивания по кускам: два движка из bench/ одной встречи.

  blind_page.py каталог-bench первый второй запись.flac [--items 40]

Пословных таймкодов у большинства движков нет, поэтому они оценены равномерно внутри
куска по числу символов: на коротком фрагменте со звуком ±2 с этого хватает. Страница
и звук в ней лежат в records/ и в репозиторий не попадают.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import compare_asr as ca


def words(man, hyp):
    out = []
    for r in man:
        toks = hyp.get(r["id"], "").split()
        total = sum(len(t) + 1 for t in toks) or 1
        pos = 0
        for t in toks:
            a = r["start"] + (r["end"] - r["start"]) * pos / total
            pos += len(t) + 1
            b = r["start"] + (r["end"] - r["start"]) * pos / total
            for part in ca.norm(t).split("-"):
                if part:
                    out.append((part, a, b))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", type=Path)
    ap.add_argument("first")
    ap.add_argument("second")
    ap.add_argument("audio", type=Path)
    ap.add_argument("--items", type=int, default=40)
    args = ap.parse_args()
    man = [json.loads(x) for x in (args.bench / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    load = lambda e: {x["id"]: x["text"] for x in map(json.loads, (args.bench / f"{e}.jsonl").read_text(encoding="utf-8").splitlines()) if "id" in x}
    A, B = words(man, load(args.first)), words(man, load(args.second))
    ops, same = ca.diff_blocks(A, B)
    cands = ca.meaningful(ops, A, B)
    print(f"{args.first} / {args.second}: совпало {100 * same / max(len(A), len(B)):.1f}%, мест по существу {len(cands)}")
    out = args.bench / f"blind-{args.first}-vs-{args.second}.html"
    ca.build_page(args.audio, cands, [args.first, args.second], out, args.items, 0)


if __name__ == "__main__":
    main()
