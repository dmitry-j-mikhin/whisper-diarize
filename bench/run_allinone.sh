#!/usr/bin/env bash
# Модели «всё в одном»: VibeVoice-ASR в 4 битах на карте, затем Nemotron Transcribe на CPU.
# Всё без сети; встречи — только в records/<запись>/bench/.
cd "$(dirname "$0")"
B=~/.cache/asr-bench
BIN=$B/nsc/x/nemo-speech-nightly-linux-x86_64-cuda/bin/nemo-speech
SALM=$B/gguf/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-AD-Q2_0_M.gguf
MA="../records/2026-10-27 12-00-09/bench"; MB="../records/2026-10-29 12-00-25/bench"
AMI=$B/data/diar/ami/wav; VV=$B/out/vibevoice; S=$B/out/salm; mkdir -p "$VV" "$S" "$MA/salm" "$MB/salm"
export HF_HUB_CACHE=$B/hub
PY=$B/envs/hf/bin/python
M="./meter.py --no-net --max-ram-gb 24"
FILTER='^\[meter\]|Error|error:|Traceback|ошибка'

echo "=== $(date +%T) vibevoice: встречи и AMI"
$M --label vv-files --out "$VV/files.meter.json" -- $PY vv_whole.py --files \
  "$MB/audio.wav=$MB/salm/vv-full.json" "$MA/audio.wav=$MA/salm/vv-full.json" \
  "$AMI/ami-01.wav=$VV/ami-01.json" "$AMI/ami-02.wav=$VV/ami-02.json" \
  "$AMI/ami-03.wav=$VV/ami-03.json" "$AMI/ami-04.wav=$VV/ami-04.json" 2>&1 | grep -E "$FILTER" | cut -c1-260
for set in fleurs60 sova100; do
  echo "=== $(date +%T) vibevoice: $set"
  $M --label vv-$set --out "$VV/$set.meter.json" -- $PY vv_whole.py --manifest $B/data/clips/$set/manifest.jsonl \
    --out "$VV/$set.jsonl" 2>&1 | grep -E "$FILTER" | cut -c1-260
done

export LD_LIBRARY_PATH=$(dirname "$BIN")/../lib NEMO_SPEECH=$BIN
COMMON="--asr.salm.max_sequences 1 --asr.salm.mtp_draft 0 --asr.salm.n_batch 512 --asr.salm.encoder_max_batch 1 --asr.backend.threads 12"
for set in sova100 fleurs60; do
  echo "=== $(date +%T) salm: $set"
  $M --label salm-$set --out "$S/$set.meter.json" -- nice -n 10 $PY asr_run.py --engine nsc --model "$SALM" --device cpu \
    --batch-size 1 --nsc-args "$COMMON --asr.salm.n_ctx_per_sequence 8192 --asr.salm.max_context 8192" \
    --manifest $B/data/clips/$set/manifest.jsonl --out "$S/$set.jsonl" 2>&1 | grep -E "$FILTER" | cut -c1-260
done
LONG="$COMMON --asr.salm.n_ctx_per_sequence 40960 --asr.salm.max_context 40960"
whole() {
  echo "=== $(date +%T) salm: $2"
  $M --label salm-$2 --out "$3.meter.json" -- nice -n 10 "$BIN" transcribe "$1" --model "$SALM" --language ru --device cpu \
    $LONG --format json --diarize -o "$3.json" --force 2>&1 | grep -E "$FILTER" | cut -c1-260
}
whole "$MB/audio.wav" meeting-b "$MB/salm/salm-full"
whole "$AMI/ami-02.wav" ami-02 "$S/ami-02"
echo "=== ALLINONE DONE $(date +%T)"
