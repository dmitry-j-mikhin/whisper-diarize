#!/usr/bin/env python3
"""
VibeVoice-ASR: сразу кто, когда и что. На 8 ГБ только в 4 битах. Модель грузится один раз.

  vv_whole.py --files запись.wav=вывод.json ...      # целые записи
  vv_whole.py --manifest m.jsonl --out hyp.jsonl     # короткие клипы для WER
"""
import argparse
import json
import time

import soundfile as sf
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, VibeVoiceAsrForConditionalGeneration

MODEL = "microsoft/VibeVoice-ASR-hf"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--manifest")
    ap.add_argument("--out")
    args = ap.parse_args()
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16)
    model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
        MODEL, quantization_config=quant, device_map="cuda:0")
    load_s = round(time.time() - t0, 1)

    def run(audio, fmt):
        inputs = proc.apply_transcription_request(audio=audio).to(model.device, torch.bfloat16)
        with torch.inference_mode():
            ids = model.generate(**inputs)
        return proc.decode(ids[:, inputs["input_ids"].shape[1]:], return_format=fmt)[0]

    for pair in args.files:
        audio, out = pair.rsplit("=", 1)
        t1 = time.time()
        try:
            parsed, err = run(audio, "parsed"), None
        except torch.cuda.OutOfMemoryError as e:
            parsed, err = [], str(e)[:200]
            torch.cuda.empty_cache()
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"load_s": load_s, "infer_s": round(time.time() - t1, 1),
                       "duration": sf.info(audio).duration, "error": err, "segments": parsed},
                      f, ensure_ascii=False)
        print(f"{audio[-30:]}: {time.time() - t1:.0f} с, ошибка: {err}", flush=True)

    if args.manifest:
        items = [json.loads(x) for x in open(args.manifest, encoding="utf-8")]
        t1 = time.time()
        with open(args.out, "w", encoding="utf-8") as f:
            for r in items:
                f.write(json.dumps({"id": r["id"], "text": run(r["path"], "transcription_only").strip()},
                                   ensure_ascii=False) + "\n")
            f.write(json.dumps({"_meta": {"engine": "vibevoice", "model": "VibeVoice-ASR 4-bit", "batch": 1,
                                          "load_s": load_s, "infer_s": round(time.time() - t1, 1),
                                          "audio_s": round(sum(r["dur"] for r in items), 1)}}) + "\n")


if __name__ == "__main__":
    main()
