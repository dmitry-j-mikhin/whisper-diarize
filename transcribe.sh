#!/usr/bin/env bash
# Транскрипция встречи целиком: звук из видео -> текст -> (опц.) разметка говорящих.
#
#   ./transcribe.sh "запись.webm"                       # GigaAM, русский
#   ./transcribe.sh "запись.webm" --diarize             # + кто говорит
#   ./transcribe.sh "запись.webm" --asr whisper         # whisper large-v3 (медленнее, но любой язык)
#   ./transcribe.sh "запись.webm" --diarize-only --diarize   # доразметить готовый .json
#
# Результаты: records/<имя записи>/<имя>.txt / .srt / .json, лог — там же <имя>.log.
# GPU NVIDIA (если стоит драйвер, см. nvidia-smi) подхватывается сам: сюда ставятся
# CUDA-сборки, transcribe.py выбирает устройство через --device auto.
set -euo pipefail
cd "$(dirname "$0")"
VENV=".venv"

command -v ffmpeg >/dev/null || { echo "нужен ffmpeg (dnf install ffmpeg)" >&2; exit 1; }
[[ $# -ge 1 ]] || { echo "usage: $0 <файл> [опции transcribe.py]" >&2; exit 1; }

if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[setup] создаю venv..." >&2
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q --upgrade pip
fi
# faster-whisper ставим всегда: он же запасной путь, когда GigaAM не подходит
# (другой язык, whisper-модель в -m, --initial-prompt)
"$VENV/bin/python" -c "import faster_whisper" 2>/dev/null || {
  echo "[setup] ставлю faster-whisper..." >&2
  "$VENV/bin/pip" install -q faster-whisper
}

# движок по умолчанию — GigaAM; отговорить от него может только явный --asr whisper
GIGAAM=1
[[ " $* " == *" --asr whisper "* ]] && GIGAAM=0

# GPU: ctranslate2 из PyPI грузит cuBLAS 12 (libcublas.so.12) из pip-пакета nvidia-cublas-cu12,
# torch нужен из индекса cu12x (он приносит тот же cuBLAS). cu130 не подходит: там .so.13.
have_gpu() { command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; }
TORCH_INDEX="https://download.pytorch.org/whl/cpu"
if have_gpu; then
  TORCH_INDEX="https://download.pytorch.org/whl/cu129"
  "$VENV/bin/pip" show -q nvidia-cublas-cu12 2>/dev/null || {
    echo "[setup] есть GPU: ставлю cuBLAS для faster-whisper..." >&2
    "$VENV/bin/pip" install -q "nvidia-cublas-cu12==12.9.*"
  }
fi

# pyannote нужен и для --diarize, и для GigaAM: длинную запись режет по речи её VAD.
# С ним приезжает torch — на CPU ~200 МБ, с CUDA ~3 ГБ, зато один раз
if [[ $GIGAAM == 1 || " $* " == *" --diarize "* ]]; then
  "$VENV/bin/python" -c "import importlib.metadata as m; m.version('pyannote.audio')" 2>/dev/null || {
    echo "[setup] ставлю pyannote.audio 4.x + torch ($TORCH_INDEX)..." >&2
    "$VENV/bin/pip" install -q torch torchaudio --index-url "$TORCH_INDEX"
    # pyannote 4.x -> модель по умолчанию speaker-diarization-community-1 (gated: auto,
    # надо один раз принять соглашение на её странице HF; активнее развивается, чем 3.1).
    "$VENV/bin/pip" install -q "pyannote.audio>=4"
  }
  # драйвер появился позже, чем venv: torch стоит CPU-сборкой — переставить (~3 ГБ).
  # --upgrade обязателен: без него pip считает 2.x+cpu и 2.x+cu129 одной версией
  if have_gpu && ! "$VENV/bin/python" -c "import torch, sys; sys.exit(0 if torch.version.cuda else 1)" 2>/dev/null; then
    echo "[setup] есть GPU, а torch без CUDA — переставляю torch/torchaudio из $TORCH_INDEX..." >&2
    "$VENV/bin/pip" install -q --upgrade torch torchaudio --index-url "$TORCH_INDEX"
  fi
  # pyannote спрашивает токен HF для gated-моделей; значение в лог не попадает
  [[ -z "${HF_TOKEN:-}" && -f "$HOME/.tokens" ]] && {
    . "$HOME/.tokens"; export HF_TOKEN="${HUGGING_FACE:-}"
  }
fi

# GigaAM: пакета на PyPI нет (там древний 0.1.0), ставим из гита без зависимостей —
# torch/onnxruntime уже стоят своих версий, а пины пакета утащили бы их назад
if [[ $GIGAAM == 1 ]]; then
  "$VENV/bin/python" -c "import gigaam" 2>/dev/null || {
    echo "[setup] ставлю GigaAM..." >&2
    "$VENV/bin/pip" install -q --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM.git"
    "$VENV/bin/pip" install -q hydra-core sentencepiece soundfile
  }
fi

src="$1"; shift
stem="$(basename "${src%.*}")"

# всё про одну запись живёт в records/<имя записи>/: исходник, txt/srt/json, лог,
# speakers.json, страницы сравнения. Записи из корня проекта забираем внутрь, чужие
# (из ~/Видео и прочего) не трогаем — туда только не пишем, результат всё равно в records/
parent="$(dirname "$(readlink -f "$src")")"
if [[ "$parent" == "$PWD/records/"* ]]; then
  dir="$parent"
else
  dir="$PWD/records/$stem"
  mkdir -p "$dir"
  if [[ "$parent" == "$PWD" ]]; then
    mv -n "$src" "$dir/"
    src="$dir/$(basename "$src")"
    echo "[setup] запись перенесена в records/$stem/" >&2
  fi
fi
# свой -o уважаем: тогда и лог кладём рядом с ним
outdir="$dir"; own_o=(-o "$dir")
for ((i = 1; i <= $#; i++)); do
  case "${!i}" in
    -o|--outdir) j=$((i + 1)); outdir="${!j}"; own_o=(); break ;;
  esac
done
log="$outdir/$stem.log"
mkdir -p "$outdir"
echo "[run] лог: $log  (следить: tail -f \"$log\")" >&2
"$VENV/bin/python" transcribe.py "$src" "${own_o[@]}" "$@" 2> >(tee -a "$log" >&2)
