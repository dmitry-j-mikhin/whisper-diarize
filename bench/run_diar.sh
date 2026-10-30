#!/usr/bin/env bash
# Очередь диаризации: pyannote и Nemotron на открытых наборах с эталоном и на встречах.
cd "$(dirname "$0")"
B=~/.cache/asr-bench
BIN=$B/nsc/x/nemo-speech-nightly-linux-x86_64-cuda/bin/nemo-speech
one() {  # метка wav каталог-вывода
  local label=$1 wav=$2 O=$3 stem; stem=$(basename "$wav" .wav)
  local out="$O/$stem.$label.rttm"
  [[ -s "$out" ]] && return
  mkdir -p "$O"
  case $label in
    community-1) cmd=(../.venv/bin/python diar_run.py --model pyannote/speaker-diarization-community-1 "$wav" "$out") ;;
    pyannote-3.1) cmd=(../.venv/bin/python diar_run.py --model pyannote/speaker-diarization-3.1 "$wav" "$out") ;;
    nemotron-diar) cmd=(env LD_LIBRARY_PATH="$(dirname "$BIN")/../lib" "$BIN" diarize "$wav"
                        --model "$B/gguf/Nemotron-3-Diarization.q8_0.gguf" --device cuda --format rttm
                        --recording-id "$stem" -o "$out" --force) ;;
  esac
  ./meter.py --no-net --label "$label" --out "$out.meter.json" -- "${cmd[@]}" 2>&1 \
    | grep -E '^\[meter\]|Error|error:|Traceback' | cut -c1-260
}
for label in "$@"; do
  echo "=== $(date +%T) $label"
  for set in ami vox; do
    for wav in "$B"/data/diar/$set/wav/*.wav; do one "$label" "$wav" "$B/out/diar-$set"; done
  done
  for m in "2026-10-29 12-00-25" "2026-10-27 12-00-09"; do
    one "$label" "../records/$m/bench/audio.wav" "../records/$m/bench/diar"
  done
done
echo "=== DIAR DONE $(date +%T)"
