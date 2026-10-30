#!/usr/bin/env python3
"""
Режет запись встречи по речи на те же куски до 30 с, что и transcribe.py для GigaAM,
и пишет их wav-ами + manifest.jsonl. Все движки потом распознают одни и те же куски:
иначе разница в нарезке смешается с разницей в моделях. Запускать из .venv проекта.

  make_chunks.py запись.flac каталог
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import soundfile as sf
import transcribe as tr


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2]).resolve()
    (out / "chunks").mkdir(parents=True, exist_ok=True)
    wav = tr.extract_audio(src, out / "audio.wav")
    device = "cuda"
    with open(out / "manifest.jsonl", "w", encoding="utf-8") as mf:
        for n, (start, end, data) in enumerate(tr.gigaam_chunks(wav, device)):
            path = out / "chunks" / f"{n:05d}.wav"
            sf.write(path, data.numpy(), 16000, subtype="PCM_16")
            mf.write(json.dumps({"id": f"{n:05d}", "path": str(path), "dur": round(end - start, 3),
                                 "start": round(start, 3), "end": round(end, 3), "text": ""}) + "\n")


if __name__ == "__main__":
    main()
