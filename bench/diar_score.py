#!/usr/bin/env python3
"""
Оценка диаризации. Запускать из окружения, где есть pyannote.metrics.

  diar_score.py der ref.rttm hyp.rttm          # DER против эталона
  diar_score.py owner hyp.rttm mic.flac them.flac   # проверка по своим дорожкам

У наших встреч нет разметки, зато есть две дорожки: где на микрофоне громко, а у
собеседников тихо, говорит владелец, и наоборот. Это настоящий внешний эталон,
но только на разделение «владелец / остальные»: как поделены между собой остальные,
по нему не видно.
"""
import json
import subprocess
import sys

import numpy as np

FRAME = 0.1
QUIET, GAP = -50.0, 10.0      # дБ: тише — не речь; разница дорожек, при которой ясно, кто говорит


def read_rttm(path):
    out = []
    for line in open(path):
        p = line.split()
        if len(p) >= 8 and p[0] == "SPEAKER":
            out.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return out


def der(ref_path, hyp_path):
    from pyannote.core import Annotation, Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    def ann(turns):
        a = Annotation()
        for i, (s, e, w) in enumerate(turns):
            a[Segment(s, e), i] = w
        return a

    ref, hyp = read_rttm(ref_path), read_rttm(hyp_path)
    metric = DiarizationErrorRate(collar=0.25, skip_overlap=False)
    d = metric(ann(ref), ann(hyp), detailed=True)
    total = d["total"] or 1
    return {"der": round(100 * d["diarization error rate"], 2),
            "miss": round(100 * d["missed detection"] / total, 2),
            "fa": round(100 * d["false alarm"] / total, 2),
            "conf": round(100 * d["confusion"] / total, 2),
            "ref_speakers": len({t[2] for t in ref}), "hyp_speakers": len({t[2] for t in hyp})}


def levels(path):
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"], check=True, capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    n = int(16000 * FRAME)
    x = x[: len(x) // n * n].reshape(-1, n)
    return 10 * np.log10((x ** 2).mean(axis=1) + 1e-12)


def owner(hyp_path, mic_path, them_path):
    mic, them = levels(mic_path), levels(them_path)
    n = min(len(mic), len(them))
    mic, them = mic[:n], them[:n]
    is_owner = (mic > QUIET) & (mic - them > GAP)
    is_other = (them > QUIET) & (them - mic > GAP)
    turns = read_rttm(hyp_path)
    labels = sorted({t[2] for t in turns})
    active = np.zeros((len(labels), n), dtype=bool)
    for s, e, w in turns:
        active[labels.index(w), int(s / FRAME):int(e / FRAME)] = True
    any_label = active.any(axis=0)
    own = [(active[i] & is_owner).sum() for i in range(len(labels))]
    best = int(np.argmax(own))
    return {"hyp_speakers": len(labels),
            "owner_frames_s": round(is_owner.sum() * FRAME), "other_frames_s": round(is_other.sum() * FRAME),
            "owner_recall": round(100 * (active[best] & is_owner).sum() / max(1, is_owner.sum()), 1),
            "owner_missed": round(100 * (is_owner & ~any_label).sum() / max(1, is_owner.sum()), 1),
            "other_as_owner": round(100 * (active[best] & is_other).sum() / max(1, is_other.sum()), 1),
            "other_missed": round(100 * (is_other & ~any_label).sum() / max(1, is_other.sum()), 1),
            "owner_clusters_over_5pct": int(sum(o > 0.05 * max(1, is_owner.sum()) for o in own))}


if __name__ == "__main__":
    fn = {"der": der, "owner": owner}[sys.argv[1]]
    print(json.dumps(fn(*sys.argv[2:]), ensure_ascii=False))
