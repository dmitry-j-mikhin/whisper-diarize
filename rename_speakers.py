#!/usr/bin/env python3
"""
Подставить имена вместо меток SPEAKER_xx в готовой расшифровке.

  ./rename_speakers.py запись.json --dump          # шпаргалка: кто сколько говорил
  ./rename_speakers.py запись.json                 # применить speakers.json из той же папки
  ./rename_speakers.py запись.json --map names.json  # ...или соответствия из другого файла

names.json — это просто {"SPEAKER_00": "Иван Иванов", "SPEAKER_01": "Пётр Петров"}.
Метки, которых нет в файле, остаются как есть. Перезаписывает .txt и .srt рядом.
"""
import argparse
import collections
import json
from pathlib import Path

import transcribe as T


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file", type=Path)
    ap.add_argument("--map", type=Path,
                    help="json со словарём метка -> имя; по умолчанию speakers.json "
                         "из папки с расшифровкой")
    ap.add_argument("--dump", action="store_true", help="показать статистику по говорящим")
    args = ap.parse_args()

    data = json.loads(args.json_file.read_text(encoding="utf-8"))
    rows = data["segments"]

    if args.dump:
        talk, words, first = collections.Counter(), collections.Counter(), {}
        for r in rows:
            sp = r.get("speaker", "?")
            talk[sp] += r["end"] - r["start"]
            words[sp] += len(r["text"].split())
            first.setdefault(sp, r["start"])
        total = sum(talk.values()) or 1
        print(f"{'метка':<16}{'время':>9}{'доля':>8}{'слов':>7}   первая реплика")
        for sp, sec in talk.most_common():
            print(f"{sp:<16}{sec / 60:>6.1f} мин{100 * sec / total:>7.1f}%"
                  f"{words[sp]:>7}   [{T.hhmmss(first[sp])}]")
        return

    names_file = args.map or args.json_file.parent / "speakers.json"
    if not names_file.exists():
        ap.error(f"нет файла с именами ({names_file}); сделайте его по образцу "
                 f"speakers.example.json или запустите с --dump")

    names = json.loads(names_file.read_text(encoding="utf-8"))
    hits = collections.Counter()
    for r in rows:
        sp = r.get("speaker")
        if sp in names:
            r["speaker"] = names[sp]
            hits[sp] += 1

    meta = {k: v for k, v in data.items() if k != "segments"}
    meta["speaker_names"] = names
    paths = T.write_outputs(rows, args.json_file.parent, args.json_file.stem, meta)
    for sp, n in hits.most_common():
        print(f"  {sp} -> {names[sp]}  ({n} сегментов)")
    print("переписано:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
