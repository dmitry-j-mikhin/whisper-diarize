#!/usr/bin/env bash
# Финальные замеры на свободной машине: скорость лучших моделей повторно (гипотезы — отдельно, в rerun/),
# и VibeVoice на целых записях, каждая в своём процессе: после нехватки памяти карта не освобождается,
# поэтому ошибки в общем прогоне run_allinone.sh на файлах после первого ничего не доказывают.
#   run_final.sh <снимок whisper> <canary.nemo>
cd "$(dirname "$0")"
W=$1; C=$2
B=~/.cache/asr-bench
export NEMO_SPEECH=$B/nsc/x/nemo-speech-nightly-linux-x86_64-cuda/bin/nemo-speech
BIN=$NEMO_SPEECH; LIB=$(dirname "$BIN")/../lib
MA="../records/2026-10-27 12-00-09/bench"; MB="../records/2026-10-29 12-00-25/bench"
VV=$B/out/vibevoice; AMI=$B/data/diar/ami/wav
FILTER='^\[meter\]|Error|error:|Traceback|ошибка'

echo "=== $(date +%T) vibevoice: целые записи по одной"
vv() {  # wav json meter
  HF_HUB_CACHE=$B/hub ./meter.py --no-net --max-ram-gb 24 --label "vv-$(basename "$2" .json)" --out "$3" -- \
    $B/envs/hf/bin/python vv_whole.py --files "$1=$2" 2>&1 | grep -E "$FILTER" | cut -c1-200
}
vv "$MA/audio.wav" "$MA/salm/vv-full.json" "$MA/salm/vv-full.meter.json"
for n in 01 02 03 04; do vv "$AMI/ami-$n.wav" "$VV/ami-$n.json" "$VV/ami-$n.meter.json"; done

asr() {  # метка движок окружение модель батч [флаги]
  local label=$1 engine=$2 env=$3 model=$4 batch=$5 extra=$6 PY LD=
  case $env in
    venv) PY=../.venv/bin/python ;;
    *)    PY=$B/envs/$env/bin/python ;;
  esac
  [[ $engine == nsc ]] && LD=$LIB
  for set in "$MA" "$B/data/clips/fleurs"; do
    O="$set/rerun"; [[ $set == $B/* ]] && O=$B/out/rerun/fleurs
    mkdir -p "$O"
    echo "=== $(date +%T) $label / $(basename "$(dirname "$set")")"
    ( [[ $env == venv ]] && unset HF_HUB_CACHE || export HF_HUB_CACHE=$B/hub
      ./meter.py --no-net --label "$label" --out "$O/$label.meter.json" -- \
        env LD_LIBRARY_PATH="$LD" "$PY" asr_run.py --engine "$engine" --model "$model" --batch-size "$batch" \
        --manifest "$set/manifest.jsonl" --out "$O/$label.jsonl" $extra 2>&1 | grep -E "$FILTER" | cut -c1-260 )
  done
}
asr gigaam-v3 gigaam venv v3_e2e_rnnt 16
asr gigaam-multi gigaam venv multilingual_large_ctc 16
asr parakeet-nsc nsc hf "$B/gguf/parakeet-tdt-0.6b-v3.q8_0.gguf" 8
asr canary-1b-v2 nemo nemo "$C" 4 --fp16
asr qwen3-1.7b qwen3 hf Qwen/Qwen3-ASR-1.7B-hf 8
asr whisper-large-v3 whisper venv "$W" 1

diar() {  # метка wav каталог
  local label=$1 wav=$2 O=$3 stem cmd; stem=$(basename "$wav" .wav); mkdir -p "$O"
  local out="$O/$stem.$label.rttm"
  case $label in
    community-1) cmd=(../.venv/bin/python diar_run.py --model pyannote/speaker-diarization-community-1 "$wav" "$out") ;;
    nemotron-diar) cmd=(env LD_LIBRARY_PATH="$LIB" "$BIN" diarize "$wav" --model "$B/gguf/Nemotron-3-Diarization.q8_0.gguf"
                        --device cuda --format rttm --recording-id "$stem" -o "$out" --force) ;;
  esac
  echo "=== $(date +%T) $label / $stem"
  ./meter.py --no-net --label "$label" --out "$out.meter.json" -- "${cmd[@]}" 2>&1 | grep -E "$FILTER" | cut -c1-260
}
for label in nemotron-diar community-1; do
  diar "$label" "$MB/audio.wav" "$MB/rerun/diar"
  diar "$label" "$AMI/ami-01.wav" "$B/out/rerun/diar"
done
echo "=== FINAL DONE $(date +%T)"
