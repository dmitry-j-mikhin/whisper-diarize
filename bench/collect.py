#!/usr/bin/env python3
"""
Сводит все замеры в bench/results/*.json. В репозиторий идут только цифры: встречи зовутся
A и B, текста расшифровок, имён и списка терминов в результатах нет.

  collect.py
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr_score
import diar_score as ds
from meeting_stats import LATIN, loops
from asr_score import norm

HERE = Path(__file__).resolve().parent
RECORDS = HERE.parent / "records"
B = Path.home() / ".cache/asr-bench"
OUT = HERE / "results"
MEETINGS = {"A": "2026-10-27 12-00-09", "B": "2026-10-29 12-00-25"}
SUBSETS = {"fleurs": 60, "sova": 100}


def meter(path):
    return json.loads(Path(path).read_text()) if Path(path).exists() else {}


def clips():
    res = {}
    for name in ("fleurs", "sova"):
        man = B / "data/clips" / name / "manifest.jsonl"
        sub_man = B / "data/clips" / f"{name}{SUBSETS[name]}" / "manifest.jsonl"
        for hyp in sorted((B / "out" / name).glob("*.jsonl")):
            label = hyp.stem
            m = meter(hyp.with_suffix(".meter.json"))
            s = asr_score.score(man, hyp)
            speed = s["audio_s"] / max(0.1, s["infer_s"])
            res.setdefault(label, {})[name] = {
                **{k: s[k] for k in ("clips", "words", "wer", "wer_ci", "wer_nodigit", "cer", "empty",
                                     "batch", "load_s", "infer_s", "audio_s")},
                "speed_x": round(speed, 1), "vram_mib": m.get("vram_mib"), "rss_gib": m.get("rss_peak_gib"),
                "subset": asr_score.score(sub_man, _subset_hyp(hyp, sub_man))["wer_nodigit"]}
        for folder, label in ((B / "out/vibevoice", "vibevoice-4bit"), (B / "out/salm", "nemotron-transcribe-cpu")):
            hyp = folder / f"{name}{SUBSETS[name]}.jsonl"
            if not hyp.exists():
                continue
            s = asr_score.score(sub_man, hyp)
            m = meter(hyp.with_suffix(".meter.json"))
            res.setdefault(label, {})[name] = {
                **{k: s[k] for k in ("clips", "words", "wer", "wer_ci", "wer_nodigit", "cer", "empty",
                                     "batch", "load_s", "infer_s", "audio_s")},
                "speed_x": round(s["audio_s"] / max(0.1, s["infer_s"]), 2), "vram_mib": m.get("vram_mib"),
                "rss_gib": m.get("rss_peak_gib"), "subset": s["wer_nodigit"]}
    return res


def _subset_hyp(hyp, sub_man):
    """Гипотезы только по клипам подвыборки: сравнивать медленные модели надо на тех же клипах."""
    keep = {json.loads(x)["id"] for x in sub_man.read_text(encoding="utf-8").splitlines()}
    tmp = B / "out/_subset.jsonl"
    with open(tmp, "w", encoding="utf-8") as f:
        for line in hyp.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("id") in keep:
                f.write(line + "\n")
    return tmp


def meetings():
    terms = [x.strip().lower() for x in (RECORDS / "bench-terms.txt").read_text(encoding="utf-8").splitlines()
             if x.strip()]
    res = {}
    for key, m in MEETINGS.items():
        out = json.loads((B / f"out/meeting-{m[:10]}.json").read_text(encoding="utf-8"))
        for name, text, info in _whole(m):
            low = text.lower()
            tokens = text.split()
            found = [t for t in terms if re.search(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)]
            out["engines"][name] = {
                "words": len(norm(text)), "latin": sum(1 for t in tokens if LATIN.search(t)),
                "punct_per_100": round(100 * sum(text.count(c) for c in ".,!?;:") / max(1, len(tokens)), 1),
                "caps_pct": round(100 * sum(1 for t in tokens if t[:1].isupper()) / max(1, len(tokens)), 1),
                "loops": loops(norm(text)), "terms_found": len(found),
                "term_hits": sum(len(re.findall(rf"(?<![a-z0-9]){re.escape(t)}(?![a-z0-9])", low)) for t in terms),
                **info}
        res[key] = out
    return res


def _whole(m):
    base = RECORDS / m / "bench/salm"
    vv = base / "vv-full.json"
    if vv.exists():
        d = json.loads(vv.read_text(encoding="utf-8"))
        t = " ".join(str(s.get("Content", "")) for s in d["segments"])
        mt = meter(B / "out/vibevoice/files.meter.json")
        yield "vibevoice-4bit", t, {"infer_s": d["infer_s"], "error": d["error"], "duration": d["duration"],
                                    "vram_mib": mt.get("vram_mib"), "speakers": len({s.get("Speaker") for s in d["segments"]})}
    sa = base / "salm-full.json"
    if sa.exists():
        d = json.loads(sa.read_text(encoding="utf-8"))
        mt = meter(base / "salm-full.meter.json")
        yield "nemotron-transcribe-cpu", d.get("text", ""), {"infer_s": mt.get("wall_s"), "duration": d.get("duration"),
                                                             "vram_mib": mt.get("vram_mib"), "rss_gib": mt.get("rss_peak_gib"),
                                                             "speakers": len({w.get("speaker") for w in d.get("words", [])})}


def vv_rttm(src, dst):
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    with open(dst, "w") as f:
        for s in d["segments"]:
            f.write(f"SPEAKER x 1 {float(s['Start']):.3f} {float(s['End']) - float(s['Start']):.3f} "
                    f"<NA> <NA> spk{s['Speaker']} <NA> <NA>\n")
    return d


def salm_rttm(src, dst, gap=1.0):
    """Пословные метки -> реплики: подряд идущие слова одного говорящего без длинной паузы."""
    d = json.loads(Path(src).read_text(encoding="utf-8"))
    turns = []
    for w in d.get("words", []):
        s, e, who = float(w["start"]), float(w["end"]), w.get("speaker")
        if turns and turns[-1][2] == who and s - turns[-1][1] <= gap:
            turns[-1][1] = e
        else:
            turns.append([s, e, who])
    with open(dst, "w") as f:
        for s, e, who in turns:
            f.write(f"SPEAKER x 1 {s:.3f} {e - s:.3f} <NA> <NA> spk{who} <NA> <NA>\n")
    return d


def diar():
    res = json.loads((B / "out/diar.json").read_text(encoding="utf-8"))
    tmp = B / "out/_tmp.rttm"
    ami = {json.loads(x)["id"]: json.loads(x) for x in (B / "data/diar/ami/manifest.jsonl").read_text().splitlines()}
    for label, folder, conv in (("vibevoice-4bit", B / "out/vibevoice", vv_rttm), ("nemotron-transcribe-cpu", B / "out/salm", salm_rttm)):
        rows = []
        for rid, r in ami.items():
            src = folder / f"{rid}.json"
            if not src.exists():
                continue
            d = conv(src, tmp)
            if d.get("error"):
                rows.append({"id": rid, "error": d["error"], "dur": r["dur"]})
                continue
            row = {**ds.der(r["rttm"], tmp), "id": rid, "dur": r["dur"]}
            if "infer_s" in d:
                row["infer_s"] = d["infer_s"]
            else:
                row["wall"] = meter(folder / f"{rid}.meter.json").get("wall_s")
            rows.append(row)
        res["public"].setdefault(label, {})["ami"] = rows
        for key, m in MEETINGS.items():
            base = RECORDS / m
            src = base / "bench/salm" / ("vv-full.json" if label.startswith("vibe") else "salm-full.json")
            if src.exists():
                d = conv(src, tmp)
                if d.get("error"):
                    res["meetings"].setdefault(key, {})[label] = {"error": d["error"]}
                else:
                    res["meetings"].setdefault(key, {})[label] = ds.owner(tmp, base / f"{m}.mic.flac", base / f"{m}.them.flac")
    # насколько Nemotron Diarization расходится с нынешней community-1 на наших встречах
    for key, m in MEETINGS.items():
        d = RECORDS / m / "bench/diar"
        a, b = d / "audio.community-1.rttm", d / "audio.nemotron-diar.rttm"
        if a.exists() and b.exists():
            res["meetings"][key]["_nemotron_vs_community1"] = ds.der(a, b)
    return res


def rerun():
    """Повторные замеры скорости на свободной машине (run_final.sh)."""
    res = {"asr": {}, "diar": {}}
    for key, folder in (("A", RECORDS / MEETINGS["A"] / "bench/rerun"), ("fleurs", B / "out/rerun/fleurs")):
        for hyp in sorted(folder.glob("*.jsonl")):
            rows = hyp.read_text(encoding="utf-8").splitlines()
            meta = json.loads(rows[-1])["_meta"]
            m = meter(hyp.with_suffix(".meter.json"))
            res["asr"].setdefault(hyp.stem, {})[key] = {
                "load_s": meta["load_s"], "infer_s": meta["infer_s"], "audio_s": meta["audio_s"],
                "speed_x": round(meta["audio_s"] / max(0.1, meta["infer_s"]), 1), "vram_mib": m.get("vram_mib"),
                "load_before": m.get("load_before"), "load_after": m.get("load_after"), "exit": m.get("exit")}
    for key, folder in (("B", RECORDS / MEETINGS["B"] / "bench/rerun/diar"), ("ami-01", B / "out/rerun/diar")):
        for path in sorted(folder.glob("*.meter.json")):
            m = meter(path)
            res["diar"].setdefault(m["label"], {})[key] = {"wall_s": m["wall_s"], "vram_mib": m.get("vram_mib"),
                                                           "load_before": m.get("load_before"), "exit": m.get("exit")}
    return res


def main():
    OUT.mkdir(exist_ok=True)
    for name, fn in (("asr_clips", clips), ("asr_meetings", meetings), ("diarization", diar), ("speed_rerun", rerun)):
        (OUT / f"{name}.json").write_text(json.dumps(fn(), ensure_ascii=False, indent=1), encoding="utf-8")
        print("написал", name)


if __name__ == "__main__":
    main()
