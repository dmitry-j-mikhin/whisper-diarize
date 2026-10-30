#!/usr/bin/env bash
# Очередь распознавания: каждая модель по каждому набору, уже посчитанное пропускается.
#   run_asr.sh <метка:движок:окружение:модель:батч[:доп. флаги asr_run.py]>...
# Встречи идут без сети (meter --no-net), их гипотезы лежат в records/<запись>/bench/ —
# каталог в .gitignore, в репозиторий попадают только цифры из bench/results/.
cd "$(dirname "$0")"
B=~/.cache/asr-bench
export NEMO_SPEECH=$B/nsc/x/nemo-speech-nightly-linux-x86_64-cuda/bin/nemo-speech
SETS=("$B/data/clips/fleurs" "$B/data/clips/sova" "../records/2026-10-29 12-00-25/bench" "../records/2026-10-27 12-00-09/bench")
for spec in "$@"; do
  IFS=: read -r label engine env model batch extra <<<"$spec"
  case $env in
    venv) PY=../.venv/bin/python; unset HF_HUB_CACHE ;;   # у .venv свой кэш с pyannote и whisper
    *)    PY=$B/envs/$env/bin/python; export HF_HUB_CACHE=$B/hub ;;
  esac
  for set in "${SETS[@]}"; do
    O="$set"; [[ $set == $B/* ]] && O=$B/out/$(basename "$set")
    [[ -s "$O/$label.jsonl" ]] && continue
    mkdir -p "$O"
    LD=; [[ $engine == nsc ]] && LD=$(dirname "$NEMO_SPEECH")/../lib
    echo "=== $(date +%T) $label / $(basename "$(dirname "$set")")/$(basename "$set")"
    ./meter.py --no-net --label "$label" --out "$O/$label.meter.json" -- \
      env LD_LIBRARY_PATH="$LD" "$PY" asr_run.py --engine "$engine" --model "$model" \
      --batch-size "$batch" --manifest "$set/manifest.jsonl" --out "$O/$label.jsonl" \
      ${extra:+$extra} 2>&1 | grep -E '^\[meter\]|Error|error:|Traceback|CUDA out' | cut -c1-300
  done
done
echo "=== QUEUE DONE $(date +%T)"
