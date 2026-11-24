"""Step E24: inference latency of each encoder type on one fixed device (spec 3.8; DECISOES 26).

    python -m mo433.latency            (on the laptop, with no other load)

The device is the laptop's CPU, in fp32, with batch size LATENCY_BATCH. The Core Ultra 9 185H
is hybrid: a batch split over its performance and efficiency cores waits for the slowest ones,
so the process is pinned to the performance cores (Linux lists them in
/sys/devices/cpu_core/cpus), one CPU per physical core (hyperthreads share a core's units), and
torch runs one thread per pinned CPU. Elsewhere it uses one CPU per physical core.

Encoders: the students c16 and c32 (any row: the latency depends only on the architecture)
and the teacher, each called as in the feature extraction, on uint8 64 px images. This
includes the preprocessing inside each wrapper: the normalization, and for the teacher the
bicubic upsampling to 224 px. The input is a fixed random batch (the content does not change
the time). Each encoder gets LATENCY_WARMUP untimed batches, then LATENCY_ITERS timed one by one.

The whole measurement is done LATENCY_ROUNDS times in a row (all encoders, then all again).
The reported value is the mean over all timed batches of all rounds, and the run is flagged
when two rounds differ by more than LATENCY_MAX_ROUND_DIFF.

Writes $MO433_DATA/latency.json: per encoder, ms per batch (mean and std), ms per image and
images per second, per round and overall; plus the threads, the CPUs and the machine state.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

from mo433 import config
from mo433.io import banner, data_dir, save_json, write_meta
from mo433.student import create_student
from mo433.teacher import load_teacher

ENCODERS = tuple(f"c{c}" for c in config.WIDTHS) + ("teacher",)


def cpu_list(text: str) -> list:
    """'0-2,5' -> [0, 1, 2, 5] (the format of the CPU lists in /sys)."""
    cpus = []
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        cpus.extend(range(int(lo), int(hi or lo) + 1))
    return cpus


def performance_cpus(sys_root=Path("/sys")) -> list:
    """One CPU per physical performance core (the first of its hyperthreads) on a hybrid
    Intel CPU; one CPU per physical core of the allowed ones otherwise."""
    p_cores = sys_root / "devices" / "cpu_core" / "cpus"
    cpus = cpu_list(p_cores.read_text()) if p_cores.exists() else sorted(os.sched_getaffinity(0))
    first = set()
    for c in cpus:
        siblings = sys_root / "devices" / "system" / "cpu" / f"cpu{c}" / "topology" / "thread_siblings_list"
        first.add(min(cpu_list(siblings.read_text())) if siblings.exists() else c)
    return sorted(first)


def machine_state() -> dict:
    """What can change a CPU timing: load, frequency governor, power profile and supply."""
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    supplies = sorted(Path("/sys/class/power_supply").glob("*/online"))
    return {"loadavg": os.getloadavg(),
            "governor": read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
            "platform_profile": read("/sys/firmware/acpi/platform_profile"),
            "ac_online": any(read(p) == "1" for p in supplies) if supplies else None}


def build(name: str, pretrained: bool = True) -> torch.nn.Module:
    if name == "teacher":
        model = load_teacher("cpu", pretrained=pretrained)
    else:
        model = create_student(int(name[1:]), seed=0)
    return model.eval()


@torch.inference_mode()
def time_batches(model, x, warmup: int, iters: int) -> np.ndarray:
    """Seconds of each of ``iters`` forward passes on ``x``, after ``warmup`` untimed ones."""
    for _ in range(warmup):
        model(x)
    times = np.empty(iters)
    for i in range(iters):
        start = time.perf_counter()
        model(x)
        times[i] = time.perf_counter() - start
    return times


def summarize(times: np.ndarray, batch: int) -> dict:
    ms = 1000 * times
    return {"ms_per_batch": float(ms.mean()), "ms_per_batch_std": float(ms.std()),
            "ms_per_image": float(ms.mean() / batch),
            "images_per_second": float(batch / times.mean()), "batches": len(times)}


def measure(models: dict, x, warmup: int, iters: int, rounds: int) -> dict:
    """Per encoder: the summary of every round and of all rounds together, and the largest
    relative difference between the round means."""
    times = {name: [] for name in models}
    for r in range(rounds):
        for name, model in models.items():
            times[name].append(time_batches(model, x, warmup, iters))
            s = summarize(times[name][-1], len(x))
            print(f"  round {r + 1}  {name:<8} {s['ms_per_batch']:9.1f} ms/batch "
                  f"(± {s['ms_per_batch_std']:.1f})  {s['images_per_second']:8.0f} img/s")
    out = {}
    for name, t in times.items():
        means = [t_r.mean() for t_r in t]
        out[name] = {**summarize(np.concatenate(t), len(x)),
                     "rounds": [summarize(t_r, len(x)) for t_r in t],
                     "round_diff": float((max(means) - min(means)) / min(means))}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true",
                        help="tiny smoke test: batch 8, 1 + 2 iterations, random teacher weights")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out = Path(args.out) if args.out else data_dir() / "latency.json"
    banner("E24: inference latency (CPU, fp32)")

    cpus = performance_cpus()
    os.sched_setaffinity(0, cpus)
    threads = len(cpus)
    torch.set_num_threads(threads)
    if args.quick:
        batch, warmup, iters, rounds = 8, 1, 2, 2
    else:
        batch, warmup, iters, rounds = (config.LATENCY_BATCH, config.LATENCY_WARMUP,
                                        config.LATENCY_ITERS, config.LATENCY_ROUNDS)
    state_before = machine_state()
    print(f"  CPUs {cpus}, {threads} torch threads, batch {batch}, "
          f"{warmup} warm-up + {iters} timed batches, {rounds} rounds; "
          f"load {state_before['loadavg'][0]:.2f}, governor {state_before['governor']}, "
          f"AC {state_before['ac_online']}")
    if state_before["ac_online"] is False:
        print("  WARNING: on battery, the CPU may be throttled")

    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 256, (batch, 3, config.IMAGE_SIZE, config.IMAGE_SIZE),
                      dtype=torch.uint8, generator=g)
    models = {name: build(name, pretrained=not args.quick) for name in ENCODERS}
    start = time.perf_counter()
    results = measure(models, x, warmup, iters, rounds)
    seconds = time.perf_counter() - start

    unstable = [n for n, r in results.items() if r["round_diff"] > config.LATENCY_MAX_ROUND_DIFF]
    for name, r in results.items():
        print(f"  {name:<8} {r['ms_per_batch']:9.1f} ms/batch  {r['images_per_second']:8.0f} img/s"
              f"  rounds differ by {100 * r['round_diff']:.1f}%")
    if unstable:
        print(f"  WARNING: rounds differ by more than {100 * config.LATENCY_MAX_ROUND_DIFF:.0f}% "
              f"for {', '.join(unstable)}: measure again with the laptop idle")
    setup = {"device": "cpu", "dtype": "float32", "batch": batch, "warmup": warmup,
             "iters": iters, "rounds": rounds, "threads": threads, "cpus": cpus,
             "input": f"uint8 {config.IMAGE_SIZE} px, preprocessing inside the encoder",
             "quick": args.quick}
    save_json({**setup, "seconds": round(seconds, 1), "machine_before": state_before,
               "machine_after": machine_state(), "stable": not unstable,
               "encoders": results}, out)
    write_meta(out, **setup)
    print(f"✓ {seconds:.0f} s, written to {out}")


if __name__ == "__main__":
    main()
