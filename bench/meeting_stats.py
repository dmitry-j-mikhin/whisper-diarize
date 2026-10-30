#!/usr/bin/env python3
"""
Что можно сказать о расшифровке встречи без эталона. На выход — только числа:
текст встреч и термины в репозиторий не попадают.

  meeting_stats.py каталог-bench [термины.txt]

термины.txt — по одному в строке, латиницей. Считается, сколько их упоминаний движок
написал правильно: у GigaAM это известное слабое место. Зацикливание и выдумки на паузах
ловятся косвенно: слов в секунду больше человеческих 6 и одинаковые n-граммы подряд.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_score import edits, norm

LATIN = re.compile(r"[A-Za-z]")


def load(path):
    return {x["id"]: x["text"] for x in map(json.loads, path.read_text(encoding="utf-8").splitlines())
            if "id" in x}


def loops(words, n=4, times=3):
    """Сколько раз одна и та же n-грамма идёт подряд не меньше times раз."""
    found, i = 0, 0
    while i + n * times <= len(words):
        if all(words[i:i + n] == words[i + k * n:i + (k + 1) * n] for k in range(1, times)):
            found += 1
            i += n * times
        else:
            i += 1
    return found


def stats(hyp, durs, terms):
    text = " ".join(hyp[k] for k in sorted(hyp))
    tokens = text.split()
    words = norm(text)
    fast = sum(1 for k, t in hyp.items() if len(norm(t)) / max(0.5, durs[k]) > 6)
    low = text.lower()
    res = {"words": len(words), "latin": sum(1 for t in tokens if LATIN.search(t)),
           "digits": sum(1 for t in tokens if any(c.isdigit() for c in t)),
           "punct_per_100": round(100 * sum(text.count(c) for c in ".,!?;:") / max(1, len(tokens)), 1),
           "caps_pct": round(100 * sum(1 for t in tokens if t[:1].isupper()) / max(1, len(tokens)), 1),
           "empty_chunks": sum(1 for t in hyp.values() if not t.strip()),
           "too_fast_chunks": fast, "loops": loops(words)}
    if terms:
        res["term_hits"] = sum(len(re.findall(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)) for t in terms)
        res["terms_found"] = sum(1 for t in terms if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low))
    return res


def main():
    d = Path(sys.argv[1])
    terms = []
    if len(sys.argv) > 2:
        terms = [x.strip().lower() for x in Path(sys.argv[2]).read_text(encoding="utf-8").splitlines() if x.strip()]
    man = [json.loads(x) for x in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    durs = {r["id"]: r["dur"] for r in man}
    engines = sorted(p.stem for p in d.glob("*.jsonl") if p.stem != "manifest")
    hyps = {e: load(d / f"{e}.jsonl") for e in engines}
    out = {"speech_min": round(sum(durs.values()) / 60, 1), "terms": len(terms), "engines": {}, "pairwise_diff_pct": {}}
    for e, h in hyps.items():
        meter = json.loads((d / f"{e}.meter.json").read_text())
        meta = next((json.loads(x)["_meta"] for x in (d / f"{e}.jsonl").read_text(encoding="utf-8").splitlines()
                     if x.startswith('{"_meta"')), {})
        out["engines"][e] = {**stats(h, durs, terms), "vram_mib": meter["vram_mib"], "rss_gib": meter["rss_peak_gib"],
                             "wall_s": meter["wall_s"], "infer_s": meta.get("infer_s"), "load_s": meta.get("load_s"),
                             "batch": meta.get("batch"), "exit": meter["exit"], "no_net": meter["no_net"]}
    for i, a in enumerate(engines):
        for b in engines[i + 1:]:
            ta, tb = norm(" ".join(hyps[a][k] for k in sorted(durs))), norm(" ".join(hyps[b][k] for k in sorted(durs)))
            # блоками по кускам: иначе квадратичная правка на 10 тысячах слов считается минутами
            e = sum(edits(norm(hyps[a][k]), norm(hyps[b][k])) for k in durs)
            out["pairwise_diff_pct"][f"{a}|{b}"] = round(100 * e / max(len(ta), len(tb)), 1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
