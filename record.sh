#!/usr/bin/env bash
# Запись встречи двумя дорожками: свой микрофон и то, что слышно от остальных.
#
#   ./record.sh                      # пишем до Ctrl+C, имя записи — дата и время
#   ./record.sh "планёрка"           # своё имя
#   ./record.sh --app firefox        # «они» — потоки Firefox вместо всего выхода (на встречах не проверено)
#   ./record.sh --stream "Звонок в Яндекс Телемосте"   # «они» — только поток с таким именем:
#                                    # Firefox зовёт потоки по заголовку вкладки (на встречах не проверено)
#   ./record.sh --list               # какие есть микрофоны и выходы
#   ./record.sh --call 1             # «они» — свой виртуальный выход «Запись-звонок-1»: его
#                                    # выбирают динамиком в настройках звонка во вкладке.
#                                    # Звонков можно писать несколько сразу: --call 1, --call 2…
#                                    # Каждый — в свою запись, микрофон — в каждую.
#   ./record.sh --call 1 --stop      # остановить запись этого звонка (и --status, --toggle так же)
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
#   <имя>.out.flac    весь выход целиком — страховка при --app/--stream: если поток вкладки
#                     подвёл (молчит, отвалился), микс собирается из этой дорожки
#   <имя>.flac        смикшированное моно — вход для ./transcribe.sh (--no-mix отключает)
#
# Как устроено: два независимых приёмника pw-record — один подписан на порты микрофона,
# другой на порты выхода. Ничего не создаётся и не переключается: в наушниках всё играет
# как играло, Телемост слышит микрофон как обычно.
# Микрофон и выход по умолчанию запись ведёт за собой: подключили гарнитуру посреди
# встречи — дорожки переезжают на неё (раз в 2 с). --mic/--sink прибивают устройство.
#
# Врозь, а не одним четырёхканальным узлом — чтобы не сводить два устройства в одну
# группу PipeWire. Узел, связанный сразу с микрофоном и с выходом, склеивает их: ведущими
# становятся часы микрофона, выход из ведущего превращается в ведомого и начинает
# подстраиваться под чужой клок. Это пересогласование и слышно как треск — и у себя,
# и у собеседников (проверено: pw-top показывает, как alsa_output уходит в follower).
# Плата за независимость — дорожки начинаются не в один сэмпл: микрофонное устройство
# поднимается позже. Останавливаются они одновременно, поэтому в конце сводим по хвосту.
#
# Сводить по хвосту можно, только если дорожка непрерывна. Приёмник без источника по
# умолчанию не пишет ничего: поток вкладки пропал — и дорожка молча укоротилась. Так было
# с --stream 24–28.09: после выхода из звонка поток исчезает, сторож ждёт ещё ~10 с, и
# «они» теряла хвост 11–13 с, а микс сдвигал собеседников на столько же. Поэтому приёмники
# с node.always-process: без источника они пишут тишину (проверено: 3,00 с за 3 с против 0).
set -euo pipefail
cd "$(dirname "$0")"

for t in ffmpeg pactl pw-record pw-link pw-dump jq; do
  command -v $t >/dev/null || { echo "нужен $t (dnf install ffmpeg pulseaudio-utils pipewire-utils jq)" >&2; exit 1; }
done

# Кто сейчас пишет — в файле состояния: его читает и индикатор в панели.
# У записи звонка (--call N) файл свой: так несколько звонков пишутся параллельно.
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

# «Не беспокоить» на время записи: dunst на паузе копит уведомления и покажет их после.
# Пауза прячет в очередь и то, что уже на экране (dunst 1.12, проверено), — поэтому
# у фоновой записи она встаёт, когда своё «Пишу» отгорело: иначе оно вылезло бы снова
# после встречи. Возвращаем уровень паузы, какой был: стояла пауза руками — останется.
DND_WAS=""; DND_JOB=""
dnd_on() {   # [через сколько секунд]
  command -v dunstctl >/dev/null || return 0
  DND_WAS="$(dunstctl get-pause-level 2>/dev/null)" || { DND_WAS=""; return 0; }
  ( sleep "${1:-0}"; dunstctl set-paused true >/dev/null 2>&1 ) & DND_JOB=$!
}
dnd_off() {   # повторный вызов ничего не делает
  [[ -n "$DND_WAS" ]] || return 0
  wait "$DND_JOB" 2>/dev/null || true   # пауза ещё не встала — дождаться, а не разминуться
  dunstctl set-pause-level "$DND_WAS" >/dev/null 2>&1 || true
  DND_WAS=""
}

stop_recording() {
  is_recording || { echo "запись не идёт" >&2; return 1; }
  local pid name dir; pid="$(val pid)"; name="$(val name)"; dir="$(val dir)"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 200); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done   # ffmpeg дописывает файлы
  bar_refresh
  echo "остановлено: $dir"
  return 0
}

name=""; APP=""; STREAM=""; CALL=""; MIC=""; SINK=""; MIX=1; list=0; mode=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list) list=1; shift ;;
    --app)  APP="${2:?чей звук писать, например --app firefox}"; shift 2 ;;
    --stream) STREAM="${2:?имя потока, например --stream «Звонок в Яндекс Телемосте»}"; shift 2 ;;
    --call) CALL="${2:?номер звонка, например --call 1}"; shift 2 ;;
    --mic)  MIC="${2:?имя источника, см. --list}"; shift 2 ;;
    --sink) SINK="${2:?имя выхода, см. --list}"; shift 2 ;;
    --no-mix) MIX=0; shift ;;
    --toggle|--stop|--status|--auto) mode="${1#--}"; shift ;;
    -h|--help) sed -n '2,22p' "$0" | sed 's/^# \?//'; exit 0 ;;
    -*) echo "не знаю опцию $1 (см. --help)" >&2; exit 1 ;;
    *)  name="$1"; shift ;;
  esac
done
[[ -n "$APP" && -n "$STREAM" ]] && { echo "--app и --stream — что-то одно" >&2; exit 1; }

# Звонок в своём виртуальном выходе. Firefox зовёт потоки по заголовку вкладки, и два звонка
# SaluteJazz неразличимы: у всех потоков media.name «SaluteJazz», один pid (11.10). Поэтому
# звонку даём свой выход — null-sink «Запись-звонок-N» с петлёй в выход по умолчанию, чтобы
# было слышно, — и во вкладке выбираем его динамиком. Выход живёт и после записи: вкладка
# остаётся на нём, следующая запись подхватит. Убрать: pactl unload-module по номерам из лога.
VSINK=""
if [[ -n "$CALL" ]]; then
  [[ "$CALL" =~ ^[0-9A-Za-z]+$ ]] || { echo "--call — номер или слово латиницей: 1, 2, devops" >&2; exit 1; }
  [[ -n "$APP$STREAM" ]] && { echo "--call и --app/--stream — что-то одно" >&2; exit 1; }
  [[ "$mode" == auto ]] && { echo "--call со сторожем --auto не работает" >&2; exit 1; }
  VSINK="meetcall$CALL"
  STATE="${XDG_RUNTIME_DIR:-/tmp}/meetrec.call$CALL.state"
  LOG="${XDG_RUNTIME_DIR:-/tmp}/meetrec.call$CALL.log"
fi

# запускаем себя же в фоне, отвязав от терминала и от того, кто нас позвал (панель, сторож)
start_bg() {   # [пояснение в уведомление о старте]
  local args=()
  [[ -n "$APP"  ]] && args+=(--app  "$APP")
  [[ -n "$STREAM" ]] && args+=(--stream "$STREAM")
  [[ -n "$CALL" ]] && args+=(--call "$CALL")
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

# Заданное руками (--mic/--sink) держим всю запись, а взятое по умолчанию ведём за умолчанием:
# подключил гарнитуру посреди встречи — звук ушёл в неё, туда же переезжает и запись.
# 30.09 гарнитуру воткнули через ~12 с после старта, а дорожка «весь выход» так и осталась
# на старом выходе — 38 минут цифровой тишины (у кнопки так же пропала бы вся «они»).
MIC_PINNED=0;  [[ -n "$MIC"  ]] && MIC_PINNED=1
SINK_PINNED=0; [[ -n "$SINK" ]] && SINK_PINNED=1
[[ -n "$MIC"  ]] || MIC="$(pactl get-default-source)"
[[ -n "$SINK" ]] || SINK="$(pactl get-default-sink)"

# ---- куда пишем -----------------------------------------------------------------------
[[ -n "$name" ]] || name="$(date '+%Y-%m-%d %H-%M-%S')${CALL:+ звонок-$CALL}"
dir="records/$name"
mic_out="$dir/$name.mic.flac"; them_out="$dir/$name.them.flac"; mix_out="$dir/$name.flac"
# страховочная дорожка «весь выход» — когда «они» пишется не с выхода, а с потока приложения
BACKUP=0; [[ -n "$APP$STREAM" ]] && BACKUP=1
out_out="$dir/$name.out.flac"
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
CAP_OUT=meetcap_out
if [[ -n "$CALL" ]]; then   # у параллельных записей узлы свои, иначе relink одной снимал бы связи другой
  CAP_MIC=meetcap_c${CALL}_mic; CAP_THEM=meetcap_c${CALL}_them; CAP_OUT=meetcap_c${CALL}_out
fi

# Виртуальный выход звонка: создать, если его ещё нет (после прошлой записи он остаётся).
ensure_vsink() {
  pactl list short sinks 2>/dev/null | awk '{print $2}' | grep -qx "$VSINK" && return 0
  local m1 m2
  m1="$(pactl load-module module-null-sink sink_name="$VSINK" \
        sink_properties=device.description="Запись-звонок-$CALL")" || return 1
  # петля в выход по умолчанию — чтобы звонок было слышно; цель не задаём, она идёт за умолчанием
  m2="$(pactl load-module module-loopback source="$VSINK.monitor" source_dont_move=true latency_msec=40)" || {
    pactl unload-module "$m1" 2>/dev/null; return 1; }
  echo "[rec] создан выход «Запись-звонок-$CALL» (модули $m1 $m2): выбери его динамиком в настройках звонка"
}

# Выходные аудиопорты узлов — строка на узел, id портов по порядку каналов (FL, FR, …).
# У микрофона это capture_*, у потока приложения output_*, у выхода monitor_* — в pactl
# он зовётся «<выход>.monitor», но узел в PipeWire тот же, без суффикса.
# Цепляем по id, а не по имени «узел:порт»: имена узлов не уникальны. У Firefox каждый
# поток зовётся «Firefox», и pw-link по имени брал первый попавшийся — 24.09 дорожка «они»
# писала вкладку мессенджера вместо звонка.
# Поток — по media.name: его Firefox берёт из заголовка вкладки, так что «--stream» выбирает
# одну вкладку, а «--app» — все потоки приложения разом.
# У приложения берём только потоки вывода (Stream/Output/Audio): у потока захвата микрофона
# тоже есть выходной порт monitor_* — копия микрофона, и 24.09 так в дорожку «они» попал
# свой голос (корреляция с дорожкой «я» 0,95).
ports_of() {   # exact <имя узла> | app <кусок имени, без учёта регистра> | stream <кусок имени потока>
  pw-dump 2>/dev/null | jq -r --arg how "$1" --arg want "${2%.monitor}" '
    (map(select(.type == "PipeWire:Interface:Node"))
      | map({key: (.id | tostring), value: (.info.props // {})}) | from_entries) as $node
    | [ .[] | select(.type == "PipeWire:Interface:Port" and .info.direction == "output")
        | .info.props | select((.["format.dsp"] // "") | test("audio"))
        | $node[.["node.id"] | tostring] as $n
        | {node: .["node.id"], id: .["object.id"], ch: (.["port.id"] | tonumber),
           name: ($n["node.name"] // ""), stream: ($n["media.name"] // ""), class: ($n["media.class"] // "")} ]
    | map(select(if $how == "app"
                 then .class == "Stream/Output/Audio"
                      and (.name | ascii_downcase | contains($want | ascii_downcase))
                      and (.name | startswith("meetcap_") | not)
                 elif $how == "stream"
                 then .class == "Stream/Output/Audio" and (.stream | contains($want))
                 else .name == $want end))
    | group_by(.node)[] | sort_by(.ch) | map(.id | tostring) | join(" ")' 2>/dev/null || true
}

# Порты узла на оба входа приёмника. Моно-источник кладём в оба канала,
# чтобы после сведения в моно уровень остался прежним.
link_pair() {   # <id портов через пробел> <приёмник>
  local -a p; read -ra p <<< "$1"
  ((${#p[@]})) || return 1
  local second=0; ((${#p[@]} > 1)) && second=1
  pw-link "${p[0]}"       "$2:input_FL" 2>/dev/null || true
  pw-link "${p[$second]}" "$2:input_FR" 2>/dev/null || true
}

link_all() {   # <приёмник> exact|app|stream <имя> — каждый подходящий узел своей парой
  local ports rc=1
  while read -r ports; do link_pair "$ports" "$1" && rc=0; done < <(ports_of "$2" "$3")
  return $rc
}

unlink_all() {   # <приёмник> — снять все связи, входящие в наш узел
  local id
  for id in $(pw-dump 2>/dev/null | jq -r --arg n "$1" '
      [.[] | select(.type == "PipeWire:Interface:Node" and .info.props["node.name"] == $n) | .id] as $me
      | .[] | select(.type == "PipeWire:Interface:Link")
      | select(.info["input-node-id"] as $i | any($me[]; . == $i)) | .id' 2>/dev/null); do
    pw-link -d "$id" 2>/dev/null || true
  done
}

# Умолчание сменилось — старые связи снимаем, а не оставляем в придачу к новым: иначе
# в «я» писались бы два микрофона сразу, а в «весь выход» — и старый выход, если он заиграет.
# Новые ставит следующий relink.
follow_defaults() {
  local now
  if (( ! MIC_PINNED )); then
    now="$(pactl get-default-source 2>/dev/null)" || now=""
    if [[ -n "$now" && "$now" != "$MIC" ]]; then
      echo "[rec] $(date +%T) микрофон по умолчанию сменился: $MIC -> $now"
      unlink_all "$CAP_MIC"; MIC="$now"
    fi
  fi
  if (( ! SINK_PINNED )); then
    now="$(pactl get-default-sink 2>/dev/null)" || now=""
    if [[ -n "$now" && "$now" != "$SINK" ]]; then
      echo "[rec] $(date +%T) выход по умолчанию сменился: $SINK -> $now"
      if [[ $BACKUP == 1 ]]; then unlink_all "$CAP_OUT"; fi
      if [[ -z "$APP$STREAM$CALL" ]]; then unlink_all "$CAP_THEM"; fi   # у кнопки «они» — это выход
      SINK="$now"
    fi
  fi
  return 0
}

relink() {
  link_all "$CAP_MIC" exact "$MIC" || return 1
  if [[ $BACKUP == 1 ]]; then link_all "$CAP_OUT" exact "$SINK" || true; fi
  if   [[ -n "$CALL"   ]]; then link_all "$CAP_THEM" exact "$VSINK"
  elif [[ -n "$APP"    ]]; then link_all "$CAP_THEM" app "$APP"
  elif [[ -n "$STREAM" ]]; then link_all "$CAP_THEM" stream "$STREAM"
  else link_all "$CAP_THEM" exact "$SINK"; fi
}

# ---- запись ---------------------------------------------------------------------------
fifo_mic="$(mktemp -u "${TMPDIR:-/tmp}/meetrec.XXXXXX")"; fifo_them="$fifo_mic.them"; fifo_out="$fifo_mic.out"
mkfifo "$fifo_mic" "$fifo_them"
[[ $BACKUP == 1 ]] && mkfifo "$fifo_out"
PW_MIC=""; PW_THEM=""; PW_OUT=""; FF_MIC=""; FF_THEM=""; FF_OUT=""; WATCH=""
cleanup() {
  [[ -n "$WATCH" ]] && kill "$WATCH" 2>/dev/null || true
  dnd_off
  rm -f "$fifo_mic" "$fifo_them" "$fifo_out"
  [[ "$(val pid)" == "$$" ]] && rm -f "$STATE"       # отметку снимаем только свою
  bar_refresh
}
trap cleanup EXIT

if [[ -n "$CALL" ]]; then
  ensure_vsink || { echo "[!] не создать выход «Запись-звонок-$CALL»" >&2; exit 1; }
fi
echo "[rec] я   <- $MIC"
if   [[ -n "$CALL"   ]]; then echo "[rec] они <- выход «Запись-звонок-$CALL» ($VSINK)"
elif [[ -n "$APP"    ]]; then echo "[rec] они <- звук приложения $APP"
elif [[ -n "$STREAM" ]]; then echo "[rec] они <- поток «$STREAM»"
else echo "[rec] они <- $SINK"; fi
[[ $BACKUP == 1 ]] && echo "[rec] out <- $SINK (страховка: весь выход)"
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
  # свойства узла только латиницей без пробелов — иначе pw-record ругается на разбор;
  # always-process — чтобы без источника писалась тишина, а дорожка не укорачивалась (см. шапку)
  pw-record --target 0 -P node.name="$1" -P media.name=meetrec -P node.always-process=true \
    --channels 2 --channel-map "FL,FR" --format f32 --rate 48000 --latency 200ms - > "$2" &
}

encoder "$fifo_mic"  "$mic_out" 1; FF_MIC=$!
encoder "$fifo_them" "$them_out";   FF_THEM=$!
[[ $BACKUP == 1 ]] && { encoder "$fifo_out" "$out_out"; FF_OUT=$!; }
capture "$CAP_MIC"  "$fifo_mic";    PW_MIC=$!
capture "$CAP_THEM" "$fifo_them";   PW_THEM=$!
[[ $BACKUP == 1 ]] && { capture "$CAP_OUT" "$fifo_out"; PW_OUT=$!; }

caps="${CAP_MIC}|${CAP_THEM}"; ncap=2
[[ $BACKUP == 1 ]] && { caps+="|${CAP_OUT}"; ncap=3; }
for _ in $(seq 50); do
  n=$(pw-link -i 2>/dev/null | grep -cE "^(${caps}):input_FL\$" || true)
  [[ $n == "$ncap" ]] && break
  sleep 0.1
done
kill -0 $PW_MIC 2>/dev/null && kill -0 $PW_THEM 2>/dev/null \
  && { [[ $BACKUP == 0 ]] || kill -0 $PW_OUT 2>/dev/null; } || {
  echo "[!] pw-record не поднялся — записывать нечем" >&2
  notify "Запись встречи" "Не удалось начать: pw-record не поднялся" 5000
  kill -TERM $PW_MIC $PW_THEM $PW_OUT 2>/dev/null || true
  kill -INT $FF_MIC $FF_THEM $FF_OUT 2>/dev/null || true; wait $FF_MIC $FF_THEM $FF_OUT 2>/dev/null || true
  rm -f "$mic_out" "$them_out" "$out_out" "$mix_out"; rmdir "$dir" 2>/dev/null || true
  exit 1
}
[[ -n "$CALL" ]] || dnd_on $(( ${MEETREC_NOTIFY:-0} == 1 ? 3 : 0 ))   # у параллельных записей пауза перетиралась бы
relink || echo "[!] не вижу портов микрофона «$MIC» — проверь ./record.sh --list" >&2
if [[ -n "$APP$STREAM" ]] && ! pw-link -l 2>/dev/null | grep -q "$CAP_THEM:input_FL"; then
  echo "[!] «$APP$STREAM» сейчас молчит: подключится само, как только пойдёт звук." >&2
fi
# поток приложения пересоздаётся при перезаходе в конференцию и смене устройства,
# а устройства по умолчанию меняются, когда подключают гарнитуру
( while :; do follow_defaults; relink >/dev/null 2>&1 || true; sleep 2; done ) & WATCH=$!

# Ctrl+C: закрываем pw-record — ffmpeg получает конец потока и дописывает файлы целиком.
# Именно TERM, а не INT: у фоновой записи (--toggle) SIGINT унаследован игнорируемым
# и до pw-record не доходит, а TERM работает в обоих случаях.
# Все приёмники гасим одним kill, чтобы дорожки кончились в один момент: по их хвостам
# потом и выравниваем начала.
trap 'kill -TERM $PW_MIC $PW_THEM $PW_OUT 2>/dev/null || true' INT TERM
all_alive() { local p; for p in $FF_MIC $FF_THEM $FF_OUT; do kill -0 "$p" 2>/dev/null || return 1; done; }
while all_alive; do
  wait -n $FF_MIC $FF_THEM $FF_OUT 2>/dev/null || true
done
kill -TERM $PW_MIC $PW_THEM $PW_OUT 2>/dev/null || true
for ff in $FF_MIC $FF_THEM $FF_OUT; do while kill -0 $ff 2>/dev/null; do wait $ff 2>/dev/null || true; done; done
kill "$WATCH" 2>/dev/null || true; WATCH=""
dnd_off   # встреча кончилась: накопленное — на экран, и «готово» ниже не застрянет в очереди

# ---- сведение и итоги ------------------------------------------------------------------
len() { ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2>/dev/null || echo 0; }
level() { ffmpeg -hide_banner -nostats -i "$1" -af volumedetect -f null - 2>&1 | awk -F': ' '/mean_volume/ {print $2}'; }
# RMS каждой секунды, дБ (-inf у цифровой тишины) — строка на секунду
rms_per_sec() {
  ffmpeg -hide_banner -nostats -i "$1" \
    -af "asetnsamples=n=48000,astats=metadata=1:reset=1,ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-" \
    -f null - 2>/dev/null | awk -F= '/RMS_level/ {print $2}'
}

# Из чего сводить микс: из «они», а если поток вкладки подвёл — из страховочной «весь выход».
# Подвёл — это когда на выходе звучало, а в «они» в те же секунды была мёртвая тишина
# (поток не подцепился, отвалился, сменил имя) хотя бы 5% звучащего времени и не меньше 10 с.
# Абсолютный порог — от ложных срабатываний: новый поток подцепляется за ≤2 с (опрос relink),
# а одиночный звук уведомления на выходе на короткой записи сам по себе даёт десятки процентов.
src_out="$them_out"; src_name="они"
if [[ $BACKUP == 1 && -s "$out_out" ]]; then
  read -r lost gap < <(paste <(rms_per_sec "$out_out") <(rms_per_sec "$them_out") | awk '
    { o=($1=="-inf")?-200:$1; t=($2=="-inf"||$2=="")?-200:$2 }
    o > -50 { act++; if (t < -80) gap++ }
    END { printf "%d %d\n", act ? 100*gap/act : 0, gap }')
  echo "[rec] «они» молчала ${gap:-?} с (${lost:-?}%) из тех, когда на выходе звучало"
  if (( ${lost:-0} >= 5 && ${gap:-0} >= 10 )); then
    src_out="$out_out"; src_name="весь выход"
    echo "[!] поток «$APP$STREAM» подвёл — микс свожу из страховочной дорожки «весь выход»" >&2
  fi
fi

d_mic=$(len "$mic_out"); d_src=$(len "$src_out")
dur=$(awk -v a="${d_mic:-0}" -v b="${d_src:-0}" 'BEGIN{print (a>b)?a:b}')
# Дорожки писались независимыми потоками и начались не в один сэмпл: устройство микрофона
# поднимается дольше, чем уже играющий выход. Кончились они одновременно — всех приёмников
# погасил один kill, и благодаря always-process без провалов посередине, — поэтому разница
# длительностей и есть сдвиг начал. На него и двигаем.
off=$(awk -v a="${d_mic:-0}" -v b="${d_src:-0}" 'BEGIN{d=(a-b)*1000; printf "%d", (d<0?-d:d)+0.5}')

if [[ $MIX == 1 ]]; then
  echo "[rec] свожу микс: я + $src_name, сдвиг дорожек $off мс"
  if awk -v a="${d_mic:-0}" -v b="${d_src:-0}" 'BEGIN{exit !(a < b)}'; then
    fc="[0:a]adelay=delays=$off:all=1[s];[s][1:a]amix=inputs=2[x]"   # своя началась позже
  else
    fc="[1:a]adelay=delays=$off:all=1[s];[0:a][s]amix=inputs=2[x]"   # чужая началась позже
  fi
  # микс — дело наживное: не собрался, значит остаёмся с дорожками, они уже на диске
  ffmpeg -hide_banner -loglevel warning -nostats -n \
    -i "$mic_out" -i "$src_out" -filter_complex "$fc" \
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
tracks=("$mic_out" "$them_out"); [[ $BACKUP == 1 ]] && tracks+=("$out_out")
for f in "${tracks[@]}"; do
  lv=$(level "$f")
  printf '[rec] %-5s %6s %9s  %s\n' "$(basename "${f%.flac}" | sed 's/.*\.//')" "$(du -h "$f" | cut -f1)" \
    "$(awk -v s="$(len "$f")" 'BEGIN{printf "%d:%02d:%02d", s/3600, s%3600/60, s%60}')" "${lv:-?}"
  case "$lv" in -9[0-9]*|-inf*) echo "      ^ тишина: не то устройство? ./record.sh --list" >&2 ;; esac
done
[[ $MIX == 1 ]] && echo "[rec] дальше: ./transcribe.sh \"$mix_out\" --diarize"
notify "Запись встречи готова" "$(awk -v s="${dur:-0}" 'BEGIN{printf "%d:%02d:%02d", s/3600, s%3600/60, s%60}') — records/$name/" 5000
exit 0
