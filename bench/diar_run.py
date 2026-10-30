#!/usr/bin/env python3
"""
Диаризация pyannote в формате RTTM. Запускать из .venv проекта.

  diar_run.py --model pyannote/speaker-diarization-community-1 audio.wav out.rttm
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import transcribe as tr


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    args = ap.parse_args()
    t0 = time.time()
    turns = tr.diarize(args.audio, SimpleNamespace(
        threads=2, speakers=None, min_speakers=None, max_speakers=None,
        diar_model=args.model, device="cuda"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rid = args.audio.stem.replace(" ", "_")
    with open(args.out, "w") as f:
        for s, e, who in turns:
            f.write(f"SPEAKER {rid} 1 {s:.3f} {e - s:.3f} <NA> <NA> {who} <NA> <NA>\n")
    print(json.dumps({"infer_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
