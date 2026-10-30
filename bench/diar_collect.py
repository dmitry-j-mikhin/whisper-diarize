#!/usr/bin/env python3
"""Сводит диаризацию по наборам: DER против эталона, проверка по дорожкам на встречах, время."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diar_score as ds

B = Path.home() / ".cache/asr-bench"
RECORDS = Path(__file__).resolve().parent.parent / "records"
MEETINGS = {"A": "2026-10-27 12-00-09", "B": "2026-10-29 12-00-25"}


def speed(meter, dur):
    m = json.loads(Path(meter).read_text())
    return m["wall_s"], m["vram_mib"], m["rss_peak_gib"]


def main():
    labels = sys.argv[1:]
    out = {"public": {}, "meetings": {}}
    for label in labels:
        res = {}
        for name in ("ami", "vox"):
            rows = []
            for line in (B / "data/diar" / name / "manifest.jsonl").read_text().splitlines():
                r = json.loads(line)
                hyp = B / f"out/diar-{name}/{r['id']}.{label}.rttm"
                if not hyp.exists():
                    continue
                d = ds.der(r["rttm"], hyp)
                wall, vram, rss = speed(f"{hyp}.meter.json", r["dur"])
                rows.append({**d, "id": r["id"], "dur": r["dur"], "wall": wall, "vram": vram, "rss": rss})
            res[name] = rows
        out["public"][label] = res
        for key, m in MEETINGS.items():
            base = RECORDS / m
            hyp = base / "bench/diar" / f"audio.{label}.rttm"
            if hyp.exists():
                own = ds.owner(hyp, base / f"{m}.mic.flac", base / f"{m}.them.flac")
                wall, vram, rss = speed(f"{hyp}.meter.json", 0)
                out["meetings"].setdefault(key, {})[label] = {**own, "wall": wall, "vram": vram, "rss": rss}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
