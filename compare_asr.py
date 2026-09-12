#!/usr/bin/env python3
"""
Сравнение двух расшифровок одной записи (например whisper против GigaAM).

  ./compare_asr.py первая.json вторая.json              # цифры и крупнейшие расхождения
  ./compare_asr.py первая.json вторая.json --page       # + слепой A/B-тест на слух

Эталонной расшифровки у встреч нет, поэтому «кто прав» не вычислить: можно только
показать, где модели разошлись, и дать послушать спорные места, не зная, где чей
вариант. Страница — обычный локальный html с кусочками звука внутри, наружу
ничего не уходит; ответы лежат в localStorage браузера.
"""
import argparse
import base64
import json
import random
import re
import subprocess
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

PUNCT = re.compile(r"[^\w\s-]", re.UNICODE)
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"\d")
# слова-паразиты: расхождение только в них — не про качество распознавания.
# «да», «не», «нет» сюда не берём, там меняется смысл
FILLER = {"э", "ээ", "эм", "м", "мм", "а", "ну", "вот", "типа", "там", "как", "бы",
          "то", "это", "и", "в", "же", "ли"}


def norm(word: str) -> str:
    word = unicodedata.normalize("NFKC", word).lower().replace("ё", "е")
    return PUNCT.sub("", word).strip("-").strip()


def load(path: Path):
    """json transcribe.py -> (слова с таймкодами, сырой текст). «что-то» == «что то»."""
    meta = json.loads(path.read_text(encoding="utf-8"))
    words, raw = [], []
    for seg in meta["segments"]:
        raw.append(seg["text"].strip())
        for w in seg.get("words") or []:
            for part in norm(w["word"]).split("-"):
                if part:
                    words.append((part, float(w["start"]), float(w["end"])))
    if not words:
        sys.exit(f"{path.name}: нет пословных таймкодов, сравнивать нечего")
    return meta, words, " ".join(raw)


def title(meta: dict, path: Path) -> str:
    """Как назвать расшифровку в отчёте: «whisper large-v3», «gigaam v3_e2e_rnnt»."""
    if meta.get("model"):
        return f"{meta.get('asr', 'whisper')} {meta['model']}"
    return path.stem


def stats(name: str, meta: dict, words: list, text: str):
    tokens = text.split()
    punct = sum(text.count(c) for c in ".,!?;:—")
    caps = sum(1 for t in tokens if t[:1].isupper())
    print(f"{name}")
    print(f"  слов          {len(words)}")
    print(f"  латиница      {sum(1 for t in tokens if LATIN.search(t))}"
          f"   цифры {sum(1 for t in tokens if DIGIT.search(t))}")
    print(f"  пунктуация    {100 * punct / max(1, len(tokens)):.0f} знаков на 100 слов,"
          f" с заглавной {100 * caps / max(1, len(tokens)):.0f}%")


def diff_blocks(A, B):
    ops = SequenceMatcher(None, [w[0] for w in A], [w[0] for w in B],
                          autojunk=False).get_opcodes()
    same = sum(i2 - i1 for tag, i1, i2, _, _ in ops if tag == "equal")
    return ops, same


def meaningful(ops, A, B):
    """Расхождения, где поменялось что-то кроме «э-э» и разбивки на слова."""
    out = []
    wa, wb = [w[0] for w in A], [w[0] for w in B]
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        left, right = wa[i1:i2], wb[j1:j2]
        if [w for w in left if w not in FILLER] == [w for w in right if w not in FILLER]:
            continue
        if "".join(left) == "".join(right):        # «102 103» против «102103» — не ошибка слуха
            continue
        times = [t for _, t, _ in A[i1:i2]] + [t for _, t, _ in B[j1:j2]]
        ends = [t for _, _, t in A[i1:i2]] + [t for _, _, t in B[j1:j2]]
        if not times:
            k = min(i1, len(A) - 1)
            times, ends = [A[k][1]], [A[k][2]]
        out.append({"start": min(times), "end": max(ends),
                    "left": " ".join(left) or "—", "right": " ".join(right) or "—",
                    "ctx_l": " ".join(wa[max(0, i1 - 6):i1]), "ctx_r": " ".join(wa[i2:i2 + 6])})
    return out


def clip(audio: Path, start: float, end: float) -> str:
    """Кусок звука вокруг спорного места -> base64 opus (24 кбит/с, ~3 КБ в секунду)."""
    s = max(0.0, start - 2.0)
    e = min(max(end + 2.0, s + 4.0), s + 16.0)
    data = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{s:.2f}",
         "-t", f"{e - s:.2f}", "-i", str(audio), "-vn", "-ac", "1",
         "-c:a", "libopus", "-b:a", "24k", "-f", "ogg", "-"],
        check=True, capture_output=True).stdout
    return base64.b64encode(data).decode()


PAGE = """<!doctype html><html lang=ru><meta charset=utf-8>
<title>Слепое сравнение расшифровок</title>
<style>
 :root{color-scheme:light dark}
 body{font:16px/1.55 system-ui,sans-serif;max-width:780px;margin:0 auto;padding:16px}
 h1{font-size:20px;margin-bottom:4px} .sub{opacity:.7;font-size:14px;margin-bottom:16px}
 .card{border:1px solid #8883;border-radius:10px;padding:12px;margin:12px 0}
 .ctx{opacity:.55;font-size:14px;margin-bottom:6px}
 .var{display:flex;gap:10px;align-items:baseline;margin:6px 0}
 .var button,.both button{border:1px solid #8886;background:transparent;cursor:pointer;font:inherit;border-radius:8px}
 .var button{min-width:96px;padding:6px 10px}
 .var button.on{background:#2a7;border-color:#2a7;color:#fff}
 .txt{font-weight:600}
 audio{width:100%;margin:4px 0;height:34px}
 .both{display:flex;gap:8px;margin-top:4px}
 .both button{flex:1;padding:5px;font-size:14px;opacity:.75}
 .both button.on{background:#777;color:#fff;opacity:1}
 #score{position:sticky;top:0;background:Canvas;padding:10px 0;border-bottom:1px solid #8883;z-index:9}
 #result{border:1px solid #8883;border-radius:10px;padding:12px;margin-bottom:40px}
 table{border-collapse:collapse} td{padding:2px 12px 2px 0}
</style>
<h1>Кто расслышал правильно</h1>
<div class=sub>Послушайте фрагмент и отметьте вариант, который совпадает со звуком.
Где чья модель — скрыто, пока не нажмёте «показать результат». Ответы сохраняются в браузере.</div>
<div id=score></div>
<div id=list></div>
<button id=reveal style="margin:20px 0;padding:10px 16px;font:inherit;border-radius:8px">показать результат</button>
<div id=result hidden></div>
<script>
const ITEMS = __ITEMS__, NAMES = __NAMES__;
const KEY = "asr-ab-" + NAMES.join("|") + "-" + ITEMS.length;
let votes = JSON.parse(localStorage.getItem(KEY) || "{}");
const list = document.getElementById("list");
ITEMS.forEach(it => {
  const d = document.createElement("div"); d.className = "card";
  d.innerHTML = `<div class=ctx>${it.t} — … ${it.ctx_l} <b>[?]</b> ${it.ctx_r} …</div>
    <audio controls preload=none src="data:audio/ogg;base64,${it.audio}"></audio>
    <div class=var><button data-v=1>вариант 1</button><span class=txt>${it.v1}</span></div>
    <div class=var><button data-v=2>вариант 2</button><span class=txt>${it.v2}</span></div>
    <div class=both><button data-v=0>оба мимо</button><button data-v=3>оба сойдут</button></div>`;
  const paint = () => d.querySelectorAll("button").forEach(
      x => x.classList.toggle("on", +x.dataset.v === votes[it.n]));
  d.querySelectorAll("button").forEach(b => b.onclick = () => {
    votes[it.n] = +b.dataset.v;
    localStorage.setItem(KEY, JSON.stringify(votes));
    paint(); draw();
  });
  paint();
  list.appendChild(d);
});
function draw(){
  document.getElementById("score").textContent =
    `отмечено ${Object.keys(votes).length} из ${ITEMS.length}`;
}
draw();
document.getElementById("reveal").onclick = () => {
  let a = 0, b = 0, both = 0, none = 0;
  ITEMS.forEach(it => {
    const v = votes[it.n]; if (v === undefined) return;
    if (v === 3) both++; else if (v === 0) none++;
    else ((v === 1) === it.swap ? b++ : a++);
  });
  const r = document.getElementById("result"); r.hidden = false;
  r.innerHTML = `<table>
    <tr><td><b>${NAMES[0]}</b> прав</td><td>${a}</td></tr>
    <tr><td><b>${NAMES[1]}</b> прав</td><td>${b}</td></tr>
    <tr><td>оба верны</td><td>${both}</td></tr>
    <tr><td>оба мимо</td><td>${none}</td></tr>
    <tr><td>отмечено</td><td>${a + b + both + none} из ${ITEMS.length}</td></tr></table>`;
};
</script>
</html>"""


def build_page(audio: Path, cands, names, out: Path, count: int, seed: int):
    random.seed(seed)
    sample = sorted(random.sample(cands, min(count, len(cands))), key=lambda c: c["start"])
    items = []
    for n, c in enumerate(sample):
        swap = random.random() < 0.5              # кто «первый» — решает монетка
        mm, ss = divmod(int(c["start"]), 60)
        hh, mm = divmod(mm, 60)
        items.append({"n": n, "t": f"{hh:02d}:{mm:02d}:{ss:02d}",
                      "audio": clip(audio, c["start"], c["end"]),
                      "ctx_l": c["ctx_l"], "ctx_r": c["ctx_r"],
                      "v1": c["right"] if swap else c["left"],
                      "v2": c["left"] if swap else c["right"], "swap": swap})
    out.write_text(PAGE.replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
                       .replace("__NAMES__", json.dumps(names, ensure_ascii=False)),
                   encoding="utf-8")
    print(f"\n{out}: {len(items)} спорных мест, {out.stat().st_size / 2**20:.1f} МБ")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("first", type=Path)
    ap.add_argument("second", type=Path)
    ap.add_argument("-n", "--top", type=int, default=10,
                    help="сколько крупнейших расхождений напечатать")
    ap.add_argument("--page", nargs="?", const="", metavar="FILE",
                    help="собрать html со слепым A/B-тестом (по умолчанию <first>-vs-<second>.html)")
    ap.add_argument("--audio", type=Path, default=None,
                    help="запись для нарезки (по умолчанию source из первого json)")
    ap.add_argument("--items", type=int, default=60, help="сколько мест дать на прослушивание")
    ap.add_argument("--seed", type=int, default=0, help="выборка мест; меняйте для новой порции")
    args = ap.parse_args()

    (meta_a, A, text_a), (meta_b, B, text_b) = load(args.first), load(args.second)
    name_a, name_b = title(meta_a, args.first), title(meta_b, args.second)
    stats(name_a, meta_a, A, text_a)
    stats(name_b, meta_b, B, text_b)

    ops, same = diff_blocks(A, B)
    total = max(len(A), len(B))
    print(f"\nсовпало {same} слов из {total} ({100 * same / total:.1f}%), "
          f"расхождение {100 * (1 - same / total):.1f}%")
    cands = meaningful(ops, A, B)
    print(f"мест с расхождением {sum(1 for o in ops if o[0] != 'equal')}, "
          f"из них по существу (не «э-э» и не разбивка слов) {len(cands)}")

    for c in sorted(cands, key=lambda c: -(len(c["left"]) + len(c["right"])))[:args.top]:
        mm, ss = divmod(int(c["start"]), 60)
        hh, mm = divmod(mm, 60)
        print(f"\n[{hh:02d}:{mm:02d}:{ss:02d}] …{c['ctx_l']}…")
        print(f"  {name_a}: {c['left']}")
        print(f"  {name_b}: {c['right']}")

    if args.page is not None:
        audio = args.audio or Path(meta_a.get("source", ""))
        if not audio.exists():
            sys.exit(f"нет записи для нарезки ({audio}), укажите --audio")
        out = Path(args.page) if args.page else Path(f"{args.first.stem}-vs-{args.second.stem}.html")
        build_page(audio, cands, [name_a, name_b], out, args.items, args.seed)


if __name__ == "__main__":
    main()
