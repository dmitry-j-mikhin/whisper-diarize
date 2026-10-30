#!/usr/bin/env python3
"""
WER и CER гипотез против эталона, с 95% бутстреп-интервалом по клипам.

  asr_score.py manifest.jsonl hyp.jsonl

Нормализация одна на всех: нижний регистр, ё -> е, без пунктуации. Числа не разворачиваем:
движки, пишущие «5» вместо «пять», получают ошибку — это честно для читаемого текста,
но отдельно считается WER только по клипам без цифр (wer_nodigit).
"""
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def norm(text: str) -> list:
    text = unicodedata.normalize("NFKC", text).lower().replace("ё", "е").replace("-", " ")
    return PUNCT.sub(" ", text).split()


def edits(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score(manifest: Path, hyp: Path) -> dict:
    refs = [json.loads(x) for x in manifest.read_text(encoding="utf-8").splitlines()]
    lines = [json.loads(x) for x in hyp.read_text(encoding="utf-8").splitlines()]
    hyps = {x["id"]: x["text"] for x in lines if "id" in x}
    meta = next((x["_meta"] for x in lines if "_meta" in x), {})
    rows = []
    for r in refs:
        ref, h = norm(r["text"]), norm(hyps.get(r["id"], ""))
        rows.append((edits(ref, h), len(ref), edits(" ".join(ref), " ".join(h)),
                     max(1, len(" ".join(ref))), any(c.isdigit() for c in r["text"]),
                     not h))
    rate = lambda s: 100 * sum(x[0] for x in s) / max(1, sum(x[1] for x in s))
    rng = random.Random(0)
    boot = sorted(rate([rng.choice(rows) for _ in rows]) for _ in range(1000))
    nodig = [x for x in rows if not x[4]]
    return {"clips": len(rows), "words": sum(x[1] for x in rows),
            "wer": round(rate(rows), 2), "wer_ci": [round(boot[25], 2), round(boot[974], 2)],
            "wer_nodigit": round(rate(nodig), 2),
            "cer": round(100 * sum(x[2] for x in rows) / sum(x[3] for x in rows), 2),
            "empty": sum(1 for x in rows if x[5]), **meta}


if __name__ == "__main__":
    print(json.dumps(score(Path(sys.argv[1]), Path(sys.argv[2])), ensure_ascii=False))
