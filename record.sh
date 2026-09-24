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
#   ./record.sh --auto [опции]       # сторож: вошёл в звонок Телемоста — пишем, вышел — стоп
#
# Результат — records/<имя записи>/:
#   <имя>.mic.flac    свой голос, сырой (без эхоподавления и шумодава — их браузер делает у себя)
#   <имя>.them.flac   остальные участники
#   <имя>.flac        смикшированное моно — вход для ./transcribe.sh (--no-mix отключает)
#
# Как устроено: два независимых приёмника pw-record — один подписан на порты микрофона,
# другой на порты выхода. Ничего не создаётся и не переключается: в наушниках всё играет
# как играло, Телемост слышит микрофон как обычно.
#
# Врозь, а не одним четырёхканальным узлом — чтобы не сводить два устройства в одну
# группу PipeWire. Узел, связанный сразу с микрофоном и с выходом, склеивает их: ведущими
# становятся часы микрофона, выход из ведущего превращается в ведомого и начинает
# подстраиваться под чужой клок. Это пересогласование и слышно как треск — и у себя,
# и у собеседников (проверено: pw-top показывает, как alsa_output уходит в follower).
# Плата за независимость — дорожки начинаются не в один сэмпл: микрофонное устройство
# поднимается позже. Останавливаются они одновременно, поэтому в конце сводим по хвосту.
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
    --toggle|--stop|--status|--auto) mode="${1#--}"; shift ;;
    -h|--help) sed -n '2,14p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "не знаю опцию $1 (см. --help)" >&2; exit 1 ;;
    *)  name="$1"; shift ;;
  esac
done

# запускаем себя же в фоне, отвязав от терминала и от того, кто нас позвал (панель, сторож)
start_bg() {   # [пояснение в уведомление о старте]
  local args=()
  [[ -n "$APP"  ]] && args+=(--app  "$APP")
  [[ -n "$MIC"  ]] && args+=(--mic  "$MIC")
  [[ -n "$SINK" ]] && args+=(--sink "$SINK")
  [[ $MIX == 0  ]] && args+=(--no-mix)
  [[ -n "$name" ]] && args+=("$name")
  MEETREC_NOTIFY=1 MEETREC_WHY="${1:-}" \
    setsid "$(readlink -f "$0")" "${args[@]}" >"$LOG" 2>&1 </dev/null 8>&- 9>&- &
  for _ in $(seq 50); do is_recording && break; sleep 0.1; done
  is_recording || { echo "не удалось запустить, лог: $LOG" >&2; return 1; }
  bar_refresh; echo "пишу: $(val dir)"
}

# Сторож звонков (--auto). Firefox называет аудиопотоки по заголовку вкладки, а Телемост
# на время звонка ставит заголовок «Звонок в Яндекс Телемосте» — уже на экране проверки
# перед входом. Звонок идёт, пока такой поток есть хоть в одну сторону: воспроизведение
# живо весь звонок, даже если микрофон выключен. После выхода страница и вкладка с чатами
# зовутся просто «Яндекс Телемост» — это не звонок.
CALL_MATCH="${MEETREC_CALL:-Звонок в Яндекс Телемосте}"
in_call() {
  # через переменную, а не grep -q: при pipefail ранний выход grep роняет pactl
  # в SIGPIPE, и найденный звонок читался бы как «нет звонка»
  local s; s="$({ pactl list sink-inputs; pactl list source-outputs; } 2>/dev/null)" || true
  [[ $s == *"$CALL_MATCH"* ]]
}

watch_calls() {
  # Потоки пропадают и на пару секунд при переходе с экрана проверки во встречу —
  # звонок считаем законченным, только если его не видно grace секунд подряд.
  local grace="${MEETREC_GRACE:-10}" call=0 ours=0 lost=""
  echo "[auto] жду звонков: поток «$CALL_MATCH»"
  while :; do
    if in_call; then
      lost=""
      if (( ! call )); then
        call=1; ours=0
        echo "[auto] $(date +%T) звонок начался"
        if is_recording; then echo "[auto] запись уже идёт: $(val dir)"
        else start_bg "по звонку, остановится сама" || true; fi
      fi
      # запись, шедшая во время звонка, — наша, даже если начата руками: её и остановим.
      # Остановленную руками посреди звонка заново не начинаем — старт только на входе.
      is_recording && ours=1
    elif (( call )); then
      lost="${lost:-$SECONDS}"
      if (( SECONDS - lost >= grace )); then
        call=0; lost=""
        echo "[auto] $(date +%T) звонок кончился"
        if (( ours )) && is_recording; then stop_recording || true; fi
      fi
    fi
    sleep 2
  done
}

case "$mode" in
  auto)
    [[ -z "$name" ]] || { echo "у --auto имя не задаётся: записи называются по времени" >&2; exit 1; }
    watch_calls ;;
  status)
    if is_recording; then
      echo "recording $(( $(date +%s) - $(val started) )) $(val name)"; exit 0
    fi
    echo idle; exit 1 ;;
  stop) stop_recording; exit $? ;;
  toggle)
    if is_recording; then stop_recording; exit $?; fi
    start_bg; exit $? ;;
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
CAP_MIC=meetcap_mic    # имена наших узлов-приёмников: по одному на дорожку
CAP_THEM=meetcap_them

# Выходные порты узла: у микрофона capture_*, у потока приложения output_*, у выхода
# monitor_* — в pactl он зовётся «<выход>.monitor», но узел в PipeWire тот же, без суффикса.
ports_of() { pw-link -o 2>/dev/null | awk -v n="${1%.monitor}:" 'index($0, n)==1' | sort; }

# Порты источника на оба входа приёмника. Моно-источник кладём в оба канала,
# чтобы после сведения в моно уровень остался прежним.
link_pair() {
  local -a p; mapfile -t p < <(printf '%s' "$1" | grep -v '^$')
  ((${#p[@]})) || return 1
  local second=0; ((${#p[@]} > 1)) && second=1
  pw-link "${p[0]}"       "$2:input_FL" 2>/dev/null || true
  pw-link "${p[$second]}" "$2:input_FR" 2>/dev/null || true
}

relink() {
  link_pair "$(ports_of "$MIC")" "$CAP_MIC" || return 1
  if [[ -n "$APP" ]]; then
    local n rc=1
    while read -r n; do
      [[ "$n" == meetcap_* ]] && continue
      link_pair "$(ports_of "$n")" "$CAP_THEM" && rc=0
    done < <(pw-link -o 2>/dev/null | sed 's/:[^:]*$//' | sort -u | grep -i -- "$APP" || true)
    return $rc
  fi
  link_pair "$(ports_of "$SINK")" "$CAP_THEM"
}

# ---- запись ---------------------------------------------------------------------------
fifo_mic="$(mktemp -u "${TMPDIR:-/tmp}/meetrec.XXXXXX")"; fifo_them="$fifo_mic.them"
mkfifo "$fifo_mic" "$fifo_them"
PW_MIC=""; PW_THEM=""; FF_MIC=""; FF_THEM=""; WATCH=""
cleanup() {
  [[ -n "$WATCH" ]] && kill "$WATCH" 2>/dev/null || true
  rm -f "$fifo_mic" "$fifo_them"
  [[ "$(val pid)" == "$$" ]] && rm -f "$STATE"       # отметку снимаем только свою
  bar_refresh
}
trap cleanup EXIT

echo "[rec] я   <- $MIC"
echo "[rec] они <- ${APP:+звук приложения }${APP:-$SINK}"
echo "[rec] пишу в records/$name/ , стоп — Ctrl+C"
notify "Запись встречи" "Пишу: $name${MEETREC_WHY:+ — $MEETREC_WHY}" 2500
bar_refresh

# Дорожка = свой ffmpeg (сводит пару каналов в моно и жмёт в flac) + свой pw-record.
# --latency 200ms: просим заведомо крупные буферы. PipeWire берёт по группе минимум из
# запросов, так что большой запрос не может заставить чужое устройство молотить чаще.
encoder() {   # <fifo> <файл> [1 — показывать прогресс]
  local st=(-nostats); [[ ${3:-} == 1 ]] && st=(-stats -stats_period 60)
  ffmpeg -hide_banner -loglevel warning "${st[@]}" -n \
    -f f32le -ar 48000 -ac 2 -i "$1" -af "pan=mono|c0=0.5*c0+0.5*c1" \
    -c:a flac -sample_fmt s16 "$2" </dev/null &
}
capture() {   # <имя узла> <fifo>
  # свойства узла только латиницей без пробелов — иначе pw-record ругается на разбор
  pw-record --target 0 -P node.name="$1" -P media.name=meetrec \
    --channels 2 --channel-map "FL,FR" --format f32 --rate 48000 --latency 200ms - > "$2" &
}

encoder "$fifo_mic"  "$mic_out" 1; FF_MIC=$!
encoder "$fifo_them" "$them_out";   FF_THEM=$!
capture "$CAP_MIC"  "$fifo_mic";    PW_MIC=$!
capture "$CAP_THEM" "$fifo_them";   PW_THEM=$!

for _ in $(seq 50); do
  n=$(pw-link -i 2>/dev/null | grep -cE "^(${CAP_MIC}|${CAP_THEM}):input_FL\$" || true)
  [[ $n == 2 ]] && break
  sleep 0.1
done
kill -0 $PW_MIC 2>/dev/null && kill -0 $PW_THEM 2>/dev/null || {
  echo "[!] pw-record не поднялся — записывать нечем" >&2
  notify "Запись встречи" "Не удалось начать: pw-record не поднялся" 5000
  kill -INT $FF_MIC $FF_THEM 2>/dev/null || true; wait $FF_MIC $FF_THEM 2>/dev/null || true
  rm -f "$mic_out" "$them_out" "$mix_out"; rmdir "$dir" 2>/dev/null || true
  exit 1
}
relink || echo "[!] не вижу портов микрофона «$MIC» — проверь ./record.sh --list" >&2
if [[ -n "$APP" ]] && ! pw-link -l 2>/dev/null | grep -q "$CAP_THEM:input_FL"; then
  echo "[!] «$APP» сейчас молчит: подключится само, как только пойдёт звук." >&2
fi
# поток приложения пересоздаётся при перезаходе в конференцию и смене устройства
( while :; do relink >/dev/null 2>&1 || true; sleep 2; done ) & WATCH=$!

# Ctrl+C: закрываем pw-record — ffmpeg получает конец потока и дописывает файлы целиком.
# Именно TERM, а не INT: у фоновой записи (--toggle) SIGINT унаследован игнорируемым
# и до pw-record не доходит, а TERM работает в обоих случаях.
# Оба приёмника гасим одним kill, чтобы дорожки кончились в один момент: по их хвостам
# потом и выравниваем начала.
trap 'kill -TERM $PW_MIC $PW_THEM 2>/dev/null || true' INT TERM
while kill -0 $FF_MIC 2>/dev/null && kill -0 $FF_THEM 2>/dev/null; do
  wait -n $FF_MIC $FF_THEM 2>/dev/null || true
done
kill -TERM $PW_MIC $PW_THEM 2>/dev/null || true
for ff in $FF_MIC $FF_THEM; do while kill -0 $ff 2>/dev/null; do wait $ff 2>/dev/null || true; done; done
kill "$WATCH" 2>/dev/null || true; WATCH=""

# ---- сведение и итоги ------------------------------------------------------------------
len() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null || echo 0; }
d_mic=$(len "$mic_out"); d_them=$(len "$them_out")
dur=$(awk -v a="${d_mic:-0}" -v b="${d_them:-0}" 'BEGIN{print (a>b)?a:b}')
# Дорожки писались независимыми потоками и начались не в один сэмпл: устройство микрофона
# поднимается дольше, чем уже играющий выход. Кончились они одновременно — обоих приёмников
# погасил один kill, — поэтому разница длительностей и есть сдвиг начал. На него и двигаем.
off=$(awk -v a="${d_mic:-0}" -v b="${d_them:-0}" 'BEGIN{d=(a-b)*1000; printf "%d", (d<0?-d:d)+0.5}')

if [[ $MIX == 1 ]]; then
  echo "[rec] свожу микс, сдвиг дорожек $off мс"
  if awk -v a="${d_mic:-0}" -v b="${d_them:-0}" 'BEGIN{exit !(a < b)}'; then
    fc="[0:a]adelay=delays=$off:all=1[s];[s][1:a]amix=inputs=2[x]"   # своя началась позже
  else
    fc="[1:a]adelay=delays=$off:all=1[s];[0:a][s]amix=inputs=2[x]"   # чужая началась позже
  fi
  # микс — дело наживное: не собрался, значит остаёмся с двумя дорожками, они уже на диске
  ffmpeg -hide_banner -loglevel warning -nostats -n \
    -i "$mic_out" -i "$them_out" -filter_complex "$fc" \
    -map "[x]" -c:a flac -sample_fmt s16 "$mix_out" </dev/null \
    || { echo "[!] микс не собрался — дорожки на месте, своди руками" >&2; MIX=0; }
fi

printf '\n[rec] записано %s\n' "$(awk -v s="${dur:-0}" 'BEGIN{printf "%d:%02d:%02d", s/3600, s%3600/60, s%60}')"
if [[ $MIX == 1 ]]; then
  echo "[rec] начала дорожек разошлись на $off мс — в миксе выровнены по концу"
else
  echo "[rec] начала дорожек разошлись на $off мс — учитывай, если складываешь таймкоды"
fi

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
