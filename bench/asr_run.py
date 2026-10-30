#!/usr/bin/env python3
"""
Прогоняет один движок распознавания по manifest.jsonl и пишет гипотезы.

  asr_run.py --engine gigaam --model v3_e2e_rnnt --manifest m.jsonl --out hyp.jsonl

Движки живут в разных окружениях: gigaam и whisper в .venv проекта, qwen3 в окружении hf,
nemo в окружении nemo, nsc — бинарник из переменной NEMO_SPEECH. Импорты поэтому внутри функций.
В конце hyp.jsonl — строка {"_meta": …} с разбивкой времени: загрузка отдельно от счёта.
"""
import argparse
import json
import os
import time
from pathlib import Path


def batches(items, size):
    """Куски близкой длины — в один батч: так меньше паддинга и пик памяти предсказуем."""
    order = sorted(items, key=lambda r: r["dur"])
    return [order[i:i + size] for i in range(0, len(order), size)]


def read_wav(path):
    import soundfile as sf
    data, _ = sf.read(path, dtype="float32")
    return data


def run_gigaam(items, args):
    import gigaam
    import torch
    from gigaam.utils import AudioDataset
    from torch.utils.data import DataLoader
    model = gigaam.load_model(args.model, device="cuda", fp16_encoder=True)
    args.t_loaded = time.time()
    dtype = next(model.parameters()).dtype
    hyp = {}
    for part in batches(items, args.batch_size):
        wavs = [torch.from_numpy(read_wav(r["path"])) for r in part]
        loader = DataLoader(AudioDataset(wavs, tokenizer=None), batch_size=len(wavs),
                            shuffle=False, collate_fn=AudioDataset.collate)
        with torch.inference_mode():
            for pad, lens in loader:
                enc, enc_len = model.forward(pad.cuda().to(dtype), lens.cuda())
                for r, (text, _) in zip(part, model._decode(enc, enc_len, lens, False)):
                    hyp[r["id"]] = text.strip()
    return hyp


def run_whisper(items, args):
    from faster_whisper import WhisperModel
    model = WhisperModel(args.model, device="cuda", compute_type="float16")
    args.t_loaded = time.time()
    hyp = {}
    for r in items:
        segs, _ = model.transcribe(read_wav(r["path"]), language="ru", beam_size=5,
                                   vad_filter=False, condition_on_previous_text=False)
        hyp[r["id"]] = " ".join(s.text.strip() for s in segs)
    return hyp


def run_qwen3(items, args):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForMultimodalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map="cuda")
    args.t_loaded = time.time()
    hyp = {}
    for part in batches(items, args.batch_size):
        kw = {"prompt": [args.context] * len(part)} if args.context else {}
        inputs = proc.apply_transcription_request(audio=[read_wav(r["path"]) for r in part],
                                                  language=["Russian"] * len(part), **kw)
        inputs = inputs.to(model.device, model.dtype)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=512)
        texts = proc.decode(out[:, inputs["input_ids"].shape[1]:],
                            return_format="transcription_only")
        for r, text in zip(part, texts):
            hyp[r["id"]] = text.strip()
    return hyp


def run_nemo(items, args):
    import nemo.collections.asr as nemo_asr
    import torch
    # веса canary в .nemo — float32, 5,9 ГБ: на 8 ГБ без половинной точности не помещается,
    # и ужимать их надо до переноса на карту
    model = nemo_asr.models.ASRModel.restore_from(args.model, map_location="cpu")
    model = model.to(torch.bfloat16) if args.fp16 else model
    model = model.cuda().eval()
    args.t_loaded = time.time()
    kw = {"source_lang": "ru", "target_lang": "ru"} if "canary" in args.model else {}
    hyp = {}
    for part in batches(items, args.batch_size):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            res = model.transcribe([r["path"] for r in part], batch_size=len(part),
                                   verbose=False, **kw)
        for r, h in zip(part, res):
            hyp[r["id"]] = (h.text if hasattr(h, "text") else str(h)).strip()
    return hyp


def run_nsc(items, args):
    """Бинарник NeMo-Speech.cpp: GGUF-модели NVIDIA. Каталог кусков батчит сам."""
    import shlex
    import subprocess
    import tempfile
    args.t_loaded = time.time()
    src = Path(items[0]["path"]).parent
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [os.environ["NEMO_SPEECH"], "transcribe", str(src), "--model", args.model,
               "--language", "ru", "--device", args.device, "-c", str(args.batch_size),
               "--output-dir", tmp, "--format", "text", "--force"] + shlex.split(args.nsc_args)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        hyp = {}
        for r in items:
            out = Path(tmp) / (Path(r["path"]).stem + ".txt")
            hyp[r["id"]] = out.read_text(encoding="utf-8").strip() if out.exists() else ""
    return hyp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=["gigaam", "whisper", "qwen3", "nemo", "nsc"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda", help="nsc: устройство")
    ap.add_argument("--nsc-args", default="", help="nsc: доп. аргументы nemo-speech одной строкой")
    ap.add_argument("--context", default=None, help="qwen3: подсказка с терминами")
    args = ap.parse_args()
    items = [json.loads(x) for x in args.manifest.read_text(encoding="utf-8").splitlines()]
    t0 = time.time()
    hyp = globals()[f"run_{args.engine}"](items, args)
    t1 = time.time()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in items:
            f.write(json.dumps({"id": r["id"], "text": hyp.get(r["id"], "")}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"_meta": {
            "engine": args.engine, "model": os.path.basename(args.model), "batch": args.batch_size,
            "load_s": round(args.t_loaded - t0, 1), "infer_s": round(t1 - args.t_loaded, 1),
            "audio_s": round(sum(r["dur"] for r in items), 1)}}) + "\n")


if __name__ == "__main__":
    main()
