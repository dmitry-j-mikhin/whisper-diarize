#!/usr/bin/env python3
"""
Запускает команду и меряет её: время, пик видеопамяти и ОЗУ, нагрузку машины рядом.

  meter.py --out замер.json --label qwen3 [--no-net] -- команда аргументы...

--no-net кладёт команду в пустой сетевой namespace: туда у неё нет даже lo наружу.
Для записей встреч обязательно. Если идёт запись звонка, замер ждёт её конца: и чтобы
не мешать встрече, и чтобы чужая нагрузка не попала в цифры.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

RECORD = Path(__file__).resolve().parent.parent / "record.sh"
PAUSE = Path.home() / ".cache/asr-bench/PAUSE"   # есть файл — новые прогоны не стартуют


def recording() -> bool:
    out = subprocess.run([str(RECORD), "--status"], capture_output=True, text=True).stdout
    return out.startswith("recording")


def gpu_used() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def tree_rss(root: int) -> int:
    """Суммарный RSS дерева процессов, КиБ."""
    kids = {}
    rss = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            st = Path(f"/proc/{d}/stat").read_text()
            ppid = int(st.rsplit(")", 1)[1].split()[1])
            rss[int(d)] = int(st.rsplit(")", 1)[1].split()[21]) * 4
            kids.setdefault(ppid, []).append(int(d))
        except (OSError, ValueError, IndexError):
            continue
    total, todo = 0, [root]
    while todo:
        p = todo.pop()
        total += rss.get(p, 0)
        todo += kids.get(p, [])
    return total


def mem_available_gib() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--max-ram-gb", type=float, default=22,
                    help="потолок ОЗУ: earlyoom может убить на этой машине сторож записи, бьём сами раньше")
    ap.add_argument("--min-avail-gb", type=float, default=4,
                    help="снять прогон, если свободной памяти осталось меньше")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if args.no_net:
        cmd = ["unshare", "--user", "--map-root-user", "--net", "--"] + cmd

    while recording() or PAUSE.exists():
        print("[meter] идёт запись звонка или пауза, жду", file=sys.stderr, flush=True)
        time.sleep(30)

    base = gpu_used()
    load0 = os.getloadavg()[0]
    t0 = time.time()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    proc = subprocess.Popen(cmd, env=env)
    peak = {"gpu": base, "rss": 0, "over": False, "paused_by_call": False}

    def watch():
        while proc.poll() is None:
            peak["gpu"] = max(peak["gpu"], gpu_used())
            peak["rss"] = max(peak["rss"], tree_rss(proc.pid))
            if peak["rss"] > args.max_ram_gb * 2**20 or mem_available_gib() < args.min_avail_gb:
                peak["over"] = True
                proc.kill()
            if recording():
                peak["paused_by_call"] = True      # цифры этого прогона шумные, в отчёте помечаем
            time.sleep(0.5)

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    code = proc.wait()
    th.join(timeout=2)
    res = {"label": args.label, "exit": code, "wall_s": round(time.time() - t0, 1),
           "gpu_base_mib": base, "gpu_peak_mib": peak["gpu"], "vram_mib": peak["gpu"] - base,
           "rss_peak_gib": round(peak["rss"] / 2**20, 2), "killed_over_ram": peak["over"],
           "call_during_run": peak["paused_by_call"], "load_before": round(load0, 1),
           "load_after": round(os.getloadavg()[0], 1), "no_net": args.no_net,
           "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[meter] {json.dumps(res, ensure_ascii=False)}", file=sys.stderr, flush=True)
    sys.exit(code if code >= 0 else 1)


if __name__ == "__main__":
    main()
