#!/usr/bin/env bash
# Запись встречи двумя дорожками: свой микрофон и то, что слышно от остальных.
#
#   ./record.sh                      # пишем до Ctrl+C, имя записи — дата и время
#   ./record.sh "планёрка"           # своё имя
#   ./record.sh --app firefox        # «они» — только звук Firefox, без уведомлений и музыки
#   ./record.sh --list               # какие есть микрофоны и выходы
#
# Управление из панели (waybar) и вообще не из терминала:
#   ./record.sh --toggle [опции]     # не пишем — запустить в фоне, пишем — остановить
#   ./record.sh --stop               # остановить фоновую запись
#   ./record.sh --status             # «recording <секунд> <имя>» или «idle» (код 1)
#
# Результат — records/<имя записи>/:
#   <имя>.mic.flac    свой голос, сырой (без эхоподавления и шумодава — их браузер делает у себя)
#   <имя>.them.flac   остальные участники
#   <имя>.flac        смикшированное моно — вход для ./transcribe.sh (--no-mix отключает)
#
# Как устроено: pw-record поднимает в PipeWire четырёхканальный приёмник и подписывается
# на микрофон (каналы 1–2) и на выход звука (3–4). Ничего не создаётся и не
# переключается: в наушниках всё играет как играло, Телемост слышит микрофон как обычно.
# Обе дорожки идут одним потоком с общими часами — они начинаются в один сэмпл
# и не разъезжаются за час записи.
set -euo pipefail
cd "$(dirname "$0")"

for t in ffmpeg pactl pw-record pw-link; do
  command -v $t >/dev/null || { echo "нужен $t (dnf install ffmpeg pulseaudio-utils pipewire-utils)" >&2; exit 1; }
done

# Кто сейчас пишет — в файле состояния: его читает и индикатор в панели.
STATE="${XDG_RUNTIME_DIR:-/tmp}/meetrec.state"   # ключ=значение: pid, name, dir, started
LOG="${XDG_RUNTIME_DIR:-/tmp}/meetrec.log"

val() { awk -F= -v k="$1" '$1==k {sub(/^[^=]*=/, ""); print; exit}' "$STATE" 2>/dev/null; }

# Пишем ли прямо сейчас: pid из файла жив и это действительно наш скрипт
is_recording() {
  [[ -f "$STATE" ]] || return 1
  local pid; pid="$(val pid)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q 'record\.sh'
}

notify() {   # уведомления только у фоновой записи, в терминале они ни к чему
  [[ ${MEETREC_NOTIFY:-0} == 1 ]] && command -v notify-send >/dev/null || return 0
  notify-send -t "${3:-3000}" "$1" "$2" 2>/dev/null || true
}
bar_refresh() { pkill -RTMIN+11 waybar 2>/dev/null || true; }   # перерисовать индикатор

stop_recording() {
  is_recording || { echo "запись не идёт" >&2; return 1; }
  local pid name dir; pid="$(val pid)"; name="$(val name)"; dir="$(val dir)"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 200); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done   # ffmpeg дописывает файлы
  bar_refresh
  echo "остановлено: $dir"
  return 0
}

name=""; APP=""; MIC=""; SINK=""; MIX=1; list=0; mode=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list=1; shift ;;
    --app)  APP="${2:?чей звук писать, например --app firefox}"; shift 2 ;;
    --mic)  MIC="${2:?имя источника, см. --list}"; shift 2 ;;
    --sink) SINK="${2:?имя выхода, см. --list}"; shift 2 ;;
    --no-mix) MIX=0; shift ;;
    --toggle|--stop|--status) mode="${1#--}"; shift ;;
    -h|--help) sed -n '2,13p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "не знаю опцию $1 (см. --help)" >&2; exit 1 ;;
    *)  name="$1"; shift ;;
  esac
done

case "$mode" in
  status)
    if is_recording; then
      echo "recording $(( $(date +%s) - $(val started) )) $(val name)"; exit 0
    fi
    echo idle; exit 1 ;;
  stop) stop_recording; exit $? ;;
  toggle)
    if is_recording; then stop_recording; exit $?; fi
    # запускаем себя же в фоне, отвязав от терминала и от панели, которая нас позвала
    args=()
    [[ -n "$APP"  ]] && args+=(--app  "$APP")
    [[ -n "$MIC"  ]] && args+=(--mic  "$MIC")
    [[ -n "$SINK" ]] && args+=(--sink "$SINK")
    [[ $MIX == 0  ]] && args+=(--no-mix)
    [[ -n "$name" ]] && args+=("$name")
    MEETREC_NOTIFY=1 setsid "$(readlink -f "$0")" "${args[@]}" >"$LOG" 2>&1 </dev/null 8>&- 9>&- &
    for _ in $(seq 50); do is_recording && break; sleep 0.1; done
    is_recording || { echo "не удалось запустить, лог: $LOG" >&2; exit 1; }
    bar_refresh; echo "пишу: $(val dir)"; exit 0 ;;
esac

descr() { pactl list "$1" | awk -v n="$2" '
  $1=="Name:" {cur=$2} $1=="Description:" && cur==n {sub(/^\t*Description: /,""); print; exit}'; }

if [[ $list == 1 ]]; then
  echo "Микрофоны (--mic):"
  pactl list short sources | awk '$2 !~ /\.monitor$/ {print $2}' \
    | while read -r n; do printf "  %-68s %s\n" "$n" "$(descr sources "$n")"; done
  echo; echo "Выходы — что слышно в них, то и попадёт в дорожку «они» (--sink):"
  pactl list short sinks | awk '{print $2}' \
    | while read -r n; do printf "  %-68s %s\n" "$n" "$(descr sinks "$n")"; done
  echo; echo "Сейчас по умолчанию:"
  echo "  микрофон  $(pactl get-default-source)"
  echo "  выход     $(pactl get-default-sink)"
  exit 0
fi

[[ -n "$MIC"  ]] || MIC="$(pactl get-default-source)"
[[ -n "$SINK" ]] || SINK="$(pactl get-default-sink)"

# ---- куда пишем -----------------------------------------------------------------------
[[ -n "$name" ]] || name="$(date '+%Y-%m-%d %H-%M-%S')"
dir="records/$name"
mic_out="$dir/$name.mic.flac"; them_out="$dir/$name.them.flac"; mix_out="$dir/$name.flac"
[[ -e "$mic_out" ]] && {
  echo "уже есть $mic_out — возьми другое имя" >&2
  notify "Запись встречи" "Не начата: $name уже есть"; exit 1; }

exec 8>"$STATE.lock"
flock -w 5 8 || { echo "не дождался блокировки $STATE.lock" >&2; exit 1; }
if is_recording; then
  echo "запись уже идёт: $(val dir) (останови: $0 --stop)" >&2
  notify "Запись встречи" "Уже идёт: $(val name)"; exit 1
fi
mkdir -p "$dir"
printf 'pid=%d\nname=%s\ndir=%s\nstarted=%d\n' "$$" "$name" "$dir" "$(date +%s)" > "$STATE"
exec 8>&-

# ---- связи в PipeWire -----------------------------------------------------------------
CAP=meetcap            # имя нашего узла-приёмника

# Выходные порты узла: у микрофона capture_*, у потока приложения output_*, у выхода
# monitor_* — в pactl он зовётся «<выход>.monitor», но узел в PipeWire тот же, без суффикса.
ports_of() { pw-link -o 2>/dev/null | awk -v n="${1%.monitor}:" 'index($0, n)==1' | sort; }

# Пара портов на пару каналов приёмника. Моно-источник кладём в оба канала пары,
# чтобы после сведения в моно уровень остался прежним.
link_pair() {
  local -a p; mapfile -t p < <(printf '%s' "$1" | grep -v '^$')
  ((${#p[@]})) || return 1
  local second=0; ((${#p[@]} > 1)) && second=1
  pw-link "${p[0]}"       "$CAP:$2" 2>/dev/null || true
  pw-link "${p[$second]}" "$CAP:$3" 2>/dev/null || true
}

relink() {
  link_pair "$(ports_of "$MIC")" input_FL input_FR || return 1
  if [[ -n "$APP" ]]; then
    local n rc=1
    while read -r n; do
      [[ "$n" == "$CAP" ]] && continue
      link_pair "$(ports_of "$n")" input_RL input_RR && rc=0
    done < <(pw-link -o 2>/dev/null | sed 's/:[^:]*$//' | sort -u | grep -i -- "$APP" || true)
    return $rc
  fi
  link_pair "$(ports_of "$SINK")" input_RL input_RR
}

# ---- запись ---------------------------------------------------------------------------
fifo="$(mktemp -u "${TMPDIR:-/tmp}/meetrec.XXXXXX")"; mkfifo "$fifo"
PW=""; FF=""; WATCH=""
cleanup() {
  [[ -n "$WATCH" ]] && kill "$WATCH" 2>/dev/null || true
  rm -f "$fifo"
  [[ "$(val pid)" == "$$" ]] && rm -f "$STATE"       # отметку снимаем только свою
  bar_refresh
}
trap cleanup EXIT

echo "[rec] я   <- $MIC"
echo "[rec] они <- ${APP:+звук приложения }${APP:-$SINK}"
echo "[rec] пишу в records/$name/ , стоп — Ctrl+C"
notify "Запись встречи" "Пишу: $name" 2500
bar_refresh

# каналы 1–2 -> своя дорожка, 3–4 -> дорожка остальных; микс тем же проходом
filter="[0:a]pan=mono|c0=0.5*c0+0.5*c1[m];[0:a]pan=mono|c0=0.5*c2+0.5*c3[t]"
outs=(-map "[m]" -c:a flac -sample_fmt s16 "$mic_out"
      -map "[t]" -c:a flac -sample_fmt s16 "$them_out")
if [[ $MIX == 1 ]]; then
  filter="$filter;[m]asplit[m1][m2];[t]asplit[t1][t2];[m2][t2]amix=inputs=2[x]"
  outs=(-map "[m1]" -c:a flac -sample_fmt s16 "$mic_out"
        -map "[t1]" -c:a flac -sample_fmt s16 "$them_out"
        -map "[x]"  -c:a flac -sample_fmt s16 "$mix_out")
fi

ffmpeg -hide_banner -loglevel warning -stats -stats_period 60 -n \
  -f f32le -ar 48000 -ac 4 -i "$fifo" -filter_complex "$filter" "${outs[@]}" </dev/null &
FF=$!
# свойства узла только латиницей без пробелов — иначе pw-record ругается на разбор
pw-record --target 0 -P node.name="$CAP" -P media.name=meetrec \
  --channels 4 --channel-map "FL,FR,RL,RR" --format f32 --rate 48000 - > "$fifo" &
PW=$!

for _ in $(seq 50); do pw-link -i 2>/dev/null | grep -q "^$CAP:input_FL" && break; sleep 0.1; done
kill -0 $PW 2>/dev/null || {
  echo "[!] pw-record не поднялся — записывать нечем" >&2
  notify "Запись встречи" "Не удалось начать: pw-record не поднялся" 5000
  kill -INT $FF 2>/dev/null || true; wait $FF 2>/dev/null || true
  rm -f "$mic_out" "$them_out" "$mix_out"; rmdir "$dir" 2>/dev/null || true
  exit 1
}
relink || echo "[!] не вижу портов микрофона «$MIC» — проверь ./record.sh --list" >&2
if [[ -n "$APP" ]] && ! pw-link -l 2>/dev/null | grep -q "$CAP:input_RL"; then
  echo "[!] «$APP» сейчас молчит: подключится само, как только пойдёт звук." >&2
fi
# поток приложения пересоздаётся при перезаходе в конференцию и смене устройства
( while :; do relink >/dev/null 2>&1 || true; sleep 2; done ) & WATCH=$!

# Ctrl+C: закрываем pw-record — ffmpeg получает конец потока и дописывает файлы целиком.
# Именно TERM, а не INT: у фоновой записи (--toggle) SIGINT унаследован игнорируемым
# и до pw-record не доходит, а TERM работает в обоих случаях.
trap 'kill -TERM $PW 2>/dev/null || true' INT TERM
while kill -0 $FF 2>/dev/null; do wait $FF 2>/dev/null || true; done
kill -TERM $PW 2>/dev/null || true
kill "$WATCH" 2>/dev/null || true; WATCH=""

# ---- итоги ----------------------------------------------------------------------------
dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$mic_out" 2>/dev/null || echo 0)
printf '\n[rec] записано %s\n' "$(awk -v s="${dur:-0}" 'BEGIN{printf "%d:%02d:%02d", s/3600, s%3600/60, s%60}')"

# средний уровень: молчащая дорожка — это забытый мьют или не то устройство
for f in "$mic_out" "$them_out"; do
  lv=$(ffmpeg -hide_banner -nostats -i "$f" -af volumedetect -f null - 2>&1 \
        | awk -F': ' '/mean_volume/ {print $2}')
  printf '[rec] %-5s %6s  %s\n' "$(basename "${f%.flac}" | sed 's/.*\.//')" "$(du -h "$f" | cut -f1)" "${lv:-?}"
  case "$lv" in -9[0-9]*|-inf*) echo "      ^ тишина: не то устройство? ./record.sh --list" >&2 ;; esac
done
[[ $MIX == 1 ]] && echo "[rec] дальше: ./transcribe.sh \"$mix_out\" --diarize"
notify "Запись встречи готова" "$(awk -v s="${dur:-0}" 'BEGIN{printf "%d:%02d:%02d", s/3600, s%3600/60, s%60}') — records/$name/" 5000
exit 0
