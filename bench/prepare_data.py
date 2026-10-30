#!/usr/bin/env python3
"""
Раскладывает открытые наборы из parquet в wav 16 кГц моно + manifest.

  prepare_data.py <каталог parquet> <каталог data>

clips/<набор>/manifest.jsonl — распознавание: {"id","path","dur","text"}
diar/<набор>/manifest.jsonl  — диаризация: {"id","path","dur","rttm","speakers"}
"""
import io
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf

MAX_CLIPS = 500          # больше не нужно: доверительный интервал WER и так уже узкий


def to_16k(raw: bytes, dst: Path) -> float:
    """Любой формат из parquet -> wav 16 кГц моно, возвращает длительность."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "pipe:0",
                    "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(dst)],
                   input=raw, check=True)
    return sf.info(str(dst)).duration


def clips(parquet: Path, out: Path, text_col: str, seed=0):
    tbl = pq.read_table(parquet)
    idx = list(range(tbl.num_rows))
    random.Random(seed).shuffle(idx)
    idx = sorted(idx[:MAX_CLIPS])
    out.mkdir(parents=True, exist_ok=True)
    total = 0.0
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for n, i in enumerate(idx):
            row = tbl.slice(i, 1).to_pylist()[0]
            wav = out / "wav" / f"{n:04d}.wav"
            dur = to_16k(row["audio"]["bytes"], wav)
            total += dur
            mf.write(json.dumps({"id": f"{n:04d}", "path": str(wav), "dur": round(dur, 3),
                                 "text": row[text_col]}, ensure_ascii=False) + "\n")
    print(f"{out.name}: {len(idx)} клипов, {total / 60:.1f} мин", flush=True)


def diar(parquet: Path, out: Path, pick):
    pf = pq.ParquetFile(parquet)
    out.mkdir(parents=True, exist_ok=True)
    (out / "wav").mkdir(exist_ok=True)
    (out / "rttm").mkdir(exist_ok=True)
    n = 0
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for batch in pf.iter_batches(batch_size=1):
            row = batch.to_pylist()[0]
            spk = sorted(set(row["speakers"]))
            n += 1
            if not pick(n, len(spk)):
                continue
            rid = f"{out.name}-{n:02d}"
            wav = out / "wav" / f"{rid}.wav"
            dur = to_16k(row["audio"]["bytes"], wav)
            rttm = out / "rttm" / f"{rid}.rttm"
            with open(rttm, "w") as r:
                for s, e, who in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"]):
                    r.write(f"SPEAKER {rid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {who} <NA> <NA>\n")
            mf.write(json.dumps({"id": rid, "path": str(wav), "dur": round(dur, 2),
                                 "rttm": str(rttm), "speakers": len(spk)}) + "\n")
            print(f"  {rid}: {dur / 60:.1f} мин, говорящих {len(spk)}", flush=True)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    clips(src / "fleurs_ru_test.parquet", dst / "clips" / "fleurs", "transcription")
    clips(src / "sova_test.parquet", dst / "clips" / "sova", "transcription")
    diar(src / "ami_ihm_test_0.parquet", dst / "diar" / "ami", lambda n, k: n <= 4)
    diar(src / "vox_test_0.parquet", dst / "diar" / "vox", lambda n, k: k >= 7)


if __name__ == "__main__":
    main()
