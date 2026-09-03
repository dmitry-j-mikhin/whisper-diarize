#!/usr/bin/env python3
"""
Локальная транскрипция аудио/видео через faster-whisper + опциональная разметка
говорящих через pyannote. Считает на GPU NVIDIA, если стоит драйвер, иначе на CPU.

Пишет рядом с исходником:
  <name>.txt   — читаемый текст с таймкодами (и с говорящими, если --diarize)
  <name>.srt   — субтитры
  <name>.json  — сегменты + метаданные (для дальнейшей обработки)

Прогресс печатается в stderr, поэтому можно следить через tail -f.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".ogg", ".opus", ".flac"}


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def extract_audio(src: Path, dst: Path) -> Path:
    """Любой контейнер (webm/mp4/mkv…) -> 16 kHz mono wav, как ждут whisper и pyannote."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        log(f"[audio] уже есть: {dst.name}")
        return dst
    log(f"[audio] извлекаю дорожку -> {dst.name}")
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(dst)],
        check=True,
    )
    return dst


def ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------- диаризация

def hf_token():
    """Токен HF из окружения или из ~/.tokens (переменная HUGGING_FACE)."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    tokens = Path.home() / ".tokens"
    if tokens.exists():
        for line in tokens.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r'\s*(?:export\s+)?HUGGING_FACE\s*=\s*["\']?([^"\'\s]+)', line)
            if m:
                return m.group(1)
    return None


def load_waveform(path: Path):
    """wav (PCM 16-bit) -> (tensor [1, time], sample_rate) без torchaudio/torchcodec.

    У pyannote встроенное декодирование идёт через torchcodec, который тут не грузится;
    штатный обход из её же предупреждения — подать аудио уже в памяти.
    """
    import wave
    import numpy as np
    import torch

    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise RuntimeError(f"ожидается 16-битный wav, а тут {8 * w.getsampwidth()} бит")
        sr, ch = w.getframerate(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype("float32") / 32768.0
    data = data.reshape(-1, ch).T                      # -> (channel, time)
    return torch.from_numpy(np.ascontiguousarray(data)), sr


def patch_torch_for_pyannote3():
    """Совместимость pyannote.audio 3.x со свежими torch/torchaudio.

    torchaudio 2.11 выкинул AudioMetaData / info / list_audio_backends — pyannote их
    трогает только на путях файлового декодирования, а мы подаём аудио тензором,
    так что хватает заглушек. torch 2.6+ грузит чекпоинты с weights_only=True, а веса
    pyannote 3.x писались раньше и содержат не только тензоры; вызывающий код передаёт
    флаг явно, поэтому значение приходится перебивать.
    """
    import dataclasses
    import wave
    import torch
    import torchaudio

    @dataclasses.dataclass
    class AudioMetaData:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 16
        encoding: str = "PCM_S"

    def info(path, *a, **kw):
        with wave.open(str(path), "rb") as w:
            return AudioMetaData(w.getframerate(), w.getnframes(), w.getnchannels(),
                                 8 * w.getsampwidth(), "PCM_S")

    for name, val in (("AudioMetaData", AudioMetaData), ("info", info),
                      ("list_audio_backends", lambda: ["soundfile"])):
        if not hasattr(torchaudio, name):
            setattr(torchaudio, name, val)

    if not getattr(torch.load, "_pyannote_patched", False):
        orig = torch.load

        def load(*a, **kw):
            kw["weights_only"] = False      # локальные веса pyannote, источник доверенный
            return orig(*a, **kw)

        load._pyannote_patched = True
        torch.load = load


def cached_pipeline_config():
    """Путь к config.yaml модели 3.1 в кэше HF, если он там есть.

    pyannote.audio 4.x подменяет 'pyannote/speaker-diarization-3.1' на новый gated-репозиторий
    speaker-diarization-community-1; загрузка из локального конфига обходит эту подмену.
    """
    root = Path.home() / ".cache/huggingface/hub/models--pyannote--speaker-diarization-3.1/snapshots"
    return next(iter(sorted(root.glob("*/config.yaml"))), None) if root.exists() else None


def pick_device(requested: str) -> str:
    """auto -> cuda, если ctranslate2 видит видеокарту (нужен драйвер NVIDIA), иначе cpu."""
    if requested == "cpu":
        return "cpu"
    import ctranslate2
    if ctranslate2.get_cuda_device_count() == 0:
        if requested == "cuda":
            sys.exit("--device cuda: ctranslate2 не видит GPU — проверь nvidia-smi (драйвер загружен?)")
        return "cpu"
    return "cuda"


def preload_cuda_libs():
    """ctranslate2 ищет libcublas.so.12 через dlopen, а в venv она лежит внутри pip-пакета
    nvidia-cublas-cu12 (приезжает с torch cu12x), куда линковщик не заглядывает. Грузим сами
    с RTLD_GLOBAL — тогда dlopen по soname найдёт уже загруженную копию."""
    import ctypes
    import glob
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    for base in (spec.submodule_search_locations or []) if spec else []:
        for name in ("libcublasLt.so.12", "libcublas.so.12"):   # Lt первым: cublas от него зависит
            for so in glob.glob(f"{base}/cublas/lib/{name}"):
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout
        return out.strip().splitlines()[0]
    except Exception:
        return "GPU"


def diarize(audio: Path, args):
    """Диаризация -> [(start, end, speaker), …]. По умолчанию community-1 (pyannote 4.x)."""
    import inspect
    import torch
    import pyannote.audio as pa

    major = int((pa.__version__ or "0").split(".")[0])
    model = getattr(args, "diar_model", "pyannote/speaker-diarization-community-1")

    if major < 4:
        # старый pyannote 3.x: community-1 недоступен, откатываемся на 3.1 из кэша
        # (с заглушками совместимости для свежих torch/torchaudio)
        patch_torch_for_pyannote3()
        from pyannote.audio import Pipeline
        source = cached_pipeline_config() or "pyannote/speaker-diarization-3.1"
        model = "pyannote/speaker-diarization-3.1"
        log(f"[diar] pyannote {pa.__version__}: {model} ({'кэш' if isinstance(source, Path) else 'хаб'})")
    else:
        # pyannote 4.x: community-1 напрямую, заглушки 3.x не нужны
        from pyannote.audio import Pipeline
        source = model
        log(f"[diar] pyannote {pa.__version__}: {model}")

    torch.set_num_threads(args.threads)
    # в pyannote.audio 4.x use_auth_token переименован в token
    kw_tok = "token" if "token" in inspect.signature(Pipeline.from_pretrained).parameters \
             else "use_auth_token"
    pipe = Pipeline.from_pretrained(str(source), **{kw_tok: hf_token()})
    if pipe is None:
        raise RuntimeError("pyannote не отдал пайплайн — проверь HF-токен и доступ к модели "
                           "(для community-1 нужно принять соглашение на странице модели)")
    dev = getattr(args, "device", "cpu")
    if dev == "cuda" and not torch.cuda.is_available():
        log("[diar] torch без CUDA (CPU-сборка?) — диаризация пойдёт на CPU; "
            "transcribe.sh переставит torch из индекса cu129")
        dev = "cpu"
        torch.set_num_threads(os.cpu_count() or 4)   # «GPU-умолчание» в 2 потока тут убьёт скорость
    pipe.to(torch.device(dev))
    log(f"[diar] устройство: {dev}" + (f" ({gpu_name()})" if dev == "cuda" else ""))

    kw = {}
    if args.speakers:
        kw["num_speakers"] = args.speakers
    if args.min_speakers:
        kw["min_speakers"] = args.min_speakers
    if args.max_speakers:
        kw["max_speakers"] = args.max_speakers

    wav, sr = load_waveform(audio)
    log(f"[diar] аудио в памяти: {wav.shape[1] / sr / 60:.1f} мин @ {sr} Гц")
    t0 = time.time()

    # pyannote зовёт hook на каждом шаге; печатаем не чаще раза в 30 с, иначе лог утонет
    state = {"step": None, "logged": 0.0, "began": t0}

    def hook(step, artifact=None, file=None, total=None, completed=None):
        now = time.time()
        if step != state["step"]:
            # ETA считаем от начала текущей стадии: сегментация и эмбеддинги идут
            # с совершенно разной скоростью, общий отсчёт врал бы в разы
            state.update(step=step, logged=0.0, began=now)
        if not total or completed is None or now - state["logged"] < 30:
            return
        state["logged"] = now
        done = now - state["began"]
        eta = done * (total - completed) / completed if completed else 0
        log(f"  [{step}] {completed}/{total} ({100 * completed / total:4.1f}%)"
            f"  ETA {hhmmss(eta)}")

    ann = pipe({"waveform": wav, "sample_rate": sr}, hook=hook, **kw)
    # pyannote 4.x без legacy отдаёт объект с .speaker_diarization; 3.x — сразу Annotation
    diar = getattr(ann, "speaker_diarization", ann)
    turns = [(t.start, t.end, sp) for t, _, sp in diar.itertracks(yield_label=True)]
    who = sorted({sp for *_, sp in turns})
    log(f"[diar] {len(turns)} реплик, говорящих: {len(who)} ({', '.join(who)}), "
        f"за {hhmmss(time.time() - t0)}")
    diarize.model_used = model  # чтобы main записал в meta фактическую модель
    return turns


def speaker_for(start, end, turns):
    """Говорящий, с чьими репликами интервал пересекается дольше всего."""
    best, best_ov = None, 0.0
    for s, e, sp in turns:
        if s >= end:
            break                       # turns отсортированы, дальше только позже
        ov = min(end, e) - max(start, s)
        if ov > best_ov:
            best, best_ov = sp, ov
    return best or "SPEAKER_??"


def assign_speakers(rows, turns):
    """Грубый режим: одна метка на весь сегмент whisper."""
    for r in rows:
        r["speaker"] = speaker_for(r["start"], r["end"], turns)
    return rows


def _runs(labels):
    """Подряд идущие одинаковые метки -> [(метка, первый индекс, последний индекс), …]."""
    runs = []
    for i, sp in enumerate(labels):
        if runs and runs[-1][0] == sp:
            runs[-1][2] = i
        else:
            runs.append([sp, i, i])
    return runs


def _regroup(rows):
    """Склеить обратно куски прошлой резки — тогда перерезать можно сколько угодно раз."""
    out = []
    for r in rows:
        src = r.get("src")
        if src is not None and out and out[-1].get("src") == src:
            prev = out[-1]
            prev["end"] = r["end"]
            prev["text"] = f"{prev['text']} {r['text']}".strip()
            prev["words"] = (prev.get("words") or []) + (r.get("words") or [])
        else:
            out.append(dict(r))
    return out


def _merge(a, b):
    """Склеить две соседние строки: говорящий берётся от `a`, текст идёт по времени."""
    first, second = sorted((a, b), key=lambda r: r["start"])
    return {**a, "start": first["start"], "end": max(a["end"], b["end"]),
            "text": f"{first['text']} {second['text']}".strip(),
            "words": (first.get("words") or []) + (second.get("words") or [])}


def _absorb_orphans(rows):
    """Прилепить куски без говорящего к предыдущей строке (или к следующей, если её нет).

    Такие куски появляются в дырах между репликами pyannote: внутри своего сегмента
    приклеить их не к чему, поэтому чиним на уровне готовых строк.
    """
    out = []
    for r in rows:
        if r.get("speaker") == "SPEAKER_??" and out:
            out[-1] = _merge(out[-1], r)          # поглощает предыдущая строка
        elif out and out[-1].get("speaker") == "SPEAKER_??":
            out[-1] = _merge(r, out[-1])          # сирота в самом начале — забирает следующая
        else:
            out.append(r)
    return out


def split_by_speaker(rows, turns, min_words=2, min_dur=0.4):
    """Режет сегменты whisper по границам реплик, опираясь на пословные таймкоды.

    Сегменты whisper нарезаны по паузам и интонации, а не по говорящим, поэтому фраза,
    внутри которой собеседник перебил или подхватил, целиком уходила одному человеку.
    Здесь метка ставится каждому слову, короткие вкрапления (дрожание границ диаризации)
    поглощаются соседями, и сегмент режется по оставшимся сменам.
    """
    rows = _regroup(rows)
    out, split_count = [], 0
    for r in rows:
        words = r.get("words") or []
        if len(words) < 2:
            r["speaker"] = speaker_for(r["start"], r["end"], turns)
            out.append(r)
            continue

        labels = [speaker_for(w["start"], w["end"], turns) for w in words]

        changed = True
        while changed:
            changed = False
            runs = _runs(labels)
            if len(runs) < 2:
                break
            for k, (sp, a, b) in enumerate(runs):
                # слова в дырах между репликами (SPEAKER_??) прилепляем к соседям всегда:
                # это не смена говорящего, а просто участок, который pyannote не разметил
                if sp != "SPEAKER_??" and (b - a + 1 >= min_words
                                           or words[b]["end"] - words[a]["start"] >= min_dur):
                    continue
                donor = runs[k - 1][0] if k else runs[k + 1][0]
                for i in range(a, b + 1):
                    labels[i] = donor
                changed = True
                break

        runs = _runs(labels)
        split_count += len(runs) > 1
        for sp, a, b in runs:
            chunk = words[a:b + 1]
            out.append({"src": r.get("src", r["id"]),
                        "start": round(chunk[0]["start"], 2),
                        "end": round(chunk[-1]["end"], 2),
                        "text": "".join(w["word"] for w in chunk).strip(),
                        "speaker": sp,
                        "words": chunk})

    out = _absorb_orphans(out)
    for i, r in enumerate(out, 1):
        r["id"] = i
    log(f"[split] {len(rows)} сегментов -> {len(out)} (разрезано по говорящим: {split_count})")
    return out


# ------------------------------------------------------------------- вывод

def write_outputs(rows, outdir: Path, stem: str, meta: dict):
    """txt (склеивая подряд идущие реплики одного говорящего), srt и json."""
    srt_path, txt_path, json_path = (outdir / f"{stem}{e}" for e in (".srt", ".txt", ".json"))

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            who = f"[{r['speaker']}] " if r.get("speaker") else ""
            f.write(f"{i}\n{ts(r['start'])} --> {ts(r['end'])}\n{who}{r['text']}\n\n")

    # абзац рвём на смене говорящего, на заметной паузе и просто по длине —
    # иначе без диаризации весь текст склеивается в одну простыню
    GAP, MAX_LEN = 2.0, 45.0
    with open(txt_path, "w", encoding="utf-8") as f:
        prev, buf, start, last_end = None, [], 0.0, 0.0

        def flush():
            if buf:
                who = f"{prev}: " if prev else ""
                f.write(f"[{hhmmss(start)}] {who}{' '.join(buf)}\n\n")

        for r in rows:
            sp = r.get("speaker")
            if sp != prev or r["start"] - last_end > GAP or r["end"] - start > MAX_LEN:
                flush()
                prev, buf, start = sp, [], r["start"]
            buf.append(r["text"])
            last_end = r["end"]
        flush()

    json_path.write_text(
        json.dumps({**meta, "segments": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    return txt_path, srt_path, json_path


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="видео или аудио файл")
    ap.add_argument("-m", "--model", default="large-v3",
                    help="small|medium|large-v3|large-v3-turbo. По умолчанию large-v3 — "
                         "качественнее; ~x1.4 realtime на этом CPU против ~x2.2 у turbo")
    ap.add_argument("-l", "--language", default="ru", help="язык ('auto' — определять)")
    ap.add_argument("-o", "--outdir", type=Path, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="auto (по умолчанию): cuda, если есть GPU NVIDIA с драйвером, иначе cpu")
    ap.add_argument("--compute-type", default=None,
                    help="int8 | int8_float32 | float32 | float16 | int8_float16. "
                         "По умолчанию int8 на CPU, float16 на GPU")
    ap.add_argument("--threads", type=int, default=None,
                    help="потоков CPU (OpenMP-пул torch и ctranslate2): все ядра на CPU, 1 на GPU "
                         "(там пул только крутится вхолостую на барьерах и греет ядра)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="батчевый режим faster-whisper: N кусков за проход, 0 — последовательно. "
                         "По умолчанию 8 на GPU (вдвое быстрее, 16 не влезает в 8 ГБ) и 0 на CPU. "
                         "Куски режутся по паузам VAD, а не окнами по 30 с, см. README")
    ap.add_argument("--no-vad", action="store_true", help="не выкидывать тишину")
    ap.add_argument("--initial-prompt", default=None,
                    help="подсказка с терминами/именами — улучшает написание жаргона")
    ap.add_argument("--keep-audio", action="store_true", help="не удалять промежуточный wav")
    ap.add_argument("--diarize", action="store_true", help="размечать говорящих (pyannote)")
    ap.add_argument("--diar-model", default="pyannote/speaker-diarization-community-1",
                    help="модель диаризации: community-1 (нужен pyannote>=4, по умолчанию) "
                         "или speaker-diarization-3.1 (для старого pyannote<4)")
    ap.add_argument("--speakers", type=int, default=None, help="точное число говорящих")
    ap.add_argument("--min-speakers", type=int, default=None)
    ap.add_argument("--max-speakers", type=int, default=None)
    ap.add_argument("--no-words", action="store_true",
                    help="без пословных таймкодов (быстрее, но нельзя резать по говорящим)")
    ap.add_argument("--no-split", action="store_true",
                    help="не резать сегменты по сменам говорящего — метка на сегмент целиком")
    ap.add_argument("--reuse-turns", action="store_true",
                    help="взять границы реплик из готового .json вместо повторного прогона pyannote")
    ap.add_argument("--diarize-only", action="store_true",
                    help="не транскрибировать заново: взять готовый <name>.json и разметить говорящих")
    args = ap.parse_args()
    args.device = pick_device(args.device)
    if args.compute_type is None:
        args.compute_type = "float16" if args.device == "cuda" else "int8"
    if args.threads is None:
        # на GPU CPU-потоки почти не считают: perf показал 96% времени в gomp_barrier_wait —
        # пул OpenMP спин-ждёт между крошечными операциями. Замер: 1, 2 и 4 потока дают одно
        # и то же время (ASR 10 мин — 44–45 с, диаризация 56 мин — 161–168 с), 20 — медленнее
        args.threads = 1 if args.device == "cuda" else (os.cpu_count() or 4)
    if args.batch_size is None:
        args.batch_size = 8 if args.device == "cuda" else 0

    src = args.input.expanduser().resolve()
    if not src.exists():
        sys.exit(f"нет такого файла: {src}")
    outdir = (args.outdir or src.parent).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    stem = src.stem

    # аудио: либо исходник уже аудио, либо тянем дорожку ffmpeg-ом
    if src.suffix.lower() in AUDIO_EXT:
        audio, drop_audio = src, False
    else:
        audio = extract_audio(src, outdir / f"{stem}.wav")
        drop_audio = not args.keep_audio
    total = probe_duration(audio)

    if args.diarize_only:
        meta = json.loads((outdir / f"{stem}.json").read_text(encoding="utf-8"))
        rows = meta.pop("segments")
        log(f"[reuse] беру готовые {len(rows)} сегментов из {stem}.json")
    else:
        if args.device == "cuda":
            preload_cuda_libs()
        import ctranslate2
        # при temperature-fallback whisper сэмплирует случайно; без seed два прогона одной записи
        # расходятся на 4–6% слов (после первого такого окна сдвигается и вся дальнейшая нарезка)
        ctranslate2.set_random_seed(0)
        from faster_whisper import WhisperModel
        where = f"cuda ({gpu_name()})" if args.device == "cuda" else f"cpu, {args.threads} threads"
        log(f"[model] {args.model} / {args.compute_type} / {where}"
            + (f" / batch {args.batch_size}" if args.batch_size else ""))
        t_load = time.time()
        model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type,
                             cpu_threads=args.threads)
        log(f"[model] готова за {time.time() - t_load:.0f} c; длительность записи {hhmmss(total)}")

        opts = dict(
            language=None if args.language == "auto" else args.language,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
            vad_parameters={"min_silence_duration_ms": 500},
            condition_on_previous_text=False,   # спасает от зацикливаний на длинных записях
            initial_prompt=args.initial_prompt,
            word_timestamps=not args.no_words,  # без них не разрезать сегмент по говорящим
        )
        if args.batch_size:
            from faster_whisper import BatchedInferencePipeline
            segments, info = BatchedInferencePipeline(model).transcribe(
                str(audio), batch_size=args.batch_size, **opts)
        else:
            segments, info = model.transcribe(str(audio), **opts)
        log(f"[lang] {info.language} (p={info.language_probability:.2f})")

        t0, rows, last = time.time(), [], 0.0
        # промежуточный txt пишем сразу — чтобы можно было читать, не дожидаясь конца
        with open(outdir / f"{stem}.txt", "w", encoding="utf-8") as draft:
            for i, seg in enumerate(segments, 1):
                rows.append({"id": i, "start": round(seg.start, 2), "end": round(seg.end, 2),
                             "text": seg.text.strip(),
                             "words": [{"start": round(w.start, 2), "end": round(w.end, 2),
                                        "word": w.word} for w in (seg.words or [])]})
                draft.write(f"[{hhmmss(seg.start)}] {seg.text.strip()}\n")
                draft.flush()
                if seg.end - last >= 60:
                    last = seg.end
                    done = time.time() - t0
                    speed = seg.end / done if done else 0
                    eta = (total - seg.end) / speed if speed else 0
                    log(f"  {hhmmss(seg.end)} / {hhmmss(total)} ({100 * seg.end / total:4.1f}%)"
                        f"  x{speed:.2f} realtime  ETA {hhmmss(eta)}")
        log(f"[asr] {len(rows)} сегментов за {hhmmss(time.time() - t0)}")
        meta = {"source": str(src), "model": args.model, "language": info.language,
                "duration": total, "device": args.device, "compute_type": args.compute_type,
                "batch_size": args.batch_size}

    if args.diarize:
        saved = meta.get("turns") if args.reuse_turns else None
        if saved:
            turns = [(t["start"], t["end"], t["speaker"]) for t in saved]
            log(f"[reuse] беру {len(turns)} границ реплик из {stem}.json, pyannote не гоняю")
        else:
            turns = sorted(diarize(audio, args))
        # границы кладём в json: перерезать сегменты потом можно без пересчёта
        meta["turns"] = [{"start": round(s, 2), "end": round(e, 2), "speaker": sp}
                         for s, e, sp in turns]
        meta["diarization"] = getattr(diarize, "model_used", args.diar_model)
        if args.no_split or not any(r.get("words") for r in rows):
            if not args.no_split:
                log("[split] пословных таймкодов нет — ставлю метку на сегмент целиком")
            assign_speakers(rows, turns)
        else:
            rows = split_by_speaker(rows, turns)

    paths = write_outputs(rows, outdir, stem, meta)
    if drop_audio and audio != src:
        audio.unlink(missing_ok=True)

    words = sum(len(r["text"].split()) for r in rows)
    log(f"[done] {len(rows)} сегментов, ~{words} слов")
    for p in paths:
        log(f"       {p}")


if __name__ == "__main__":
    main()
