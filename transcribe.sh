#!/usr/bin/env bash
# Транскрипция встречи целиком: звук из видео -> текст -> (опц.) разметка говорящих.
#
#   ./transcribe.sh "запись.webm"                       # large-v3, русский
#   ./transcribe.sh "запись.webm" --diarize             # + кто говорит
#   ./transcribe.sh "запись.webm" -m large-v3-turbo     # быстрее (~x2.2 vs ~x1.4 realtime)
#   ./transcribe.sh "запись.webm" --diarize-only --diarize   # доразметить готовый .json
#
# Результаты: <имя>.txt / .srt / .json рядом с исходником, лог — <имя>.log.
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
"$VENV/bin/python" -c "import faster_whisper" 2>/dev/null || {
  echo "[setup] ставлю faster-whisper..." >&2
  "$VENV/bin/pip" install -q faster-whisper
}

# pyannote тянем только по требованию: с ним приезжает torch (~200 МБ даже в CPU-сборке)
if [[ " $* " == *" --diarize "* ]]; then
  "$VENV/bin/python" -c "import importlib.metadata as m; m.version('pyannote.audio')" 2>/dev/null || {
    echo "[setup] ставлю pyannote.audio 4.x + torch (CPU-сборка, без CUDA)..." >&2
    "$VENV/bin/pip" install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    # pyannote 4.x -> модель по умолчанию speaker-diarization-community-1 (gated: auto,
    # надо один раз принять соглашение на её странице HF; активнее развивается, чем 3.1).
    "$VENV/bin/pip" install -q "pyannote.audio>=4"
  }
  # pyannote спрашивает токен HF для gated-моделей; значение в лог не попадает
  [[ -z "${HF_TOKEN:-}" && -f "$HOME/.tokens" ]] && {
    . "$HOME/.tokens"; export HF_TOKEN="${HUGGING_FACE:-}"
  }
fi

src="$1"; shift
log="${src%.*}.log"
echo "[run] лог: $log  (следить: tail -f \"$log\")" >&2
"$VENV/bin/python" transcribe.py "$src" "$@" 2> >(tee -a "$log" >&2)
