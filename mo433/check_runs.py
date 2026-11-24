"""Step E20: check the 20 main runs of rows B and C together, before the evaluation.

    python -m mo433.check_runs

For every run of train.main_runs (rows B and C, both widths, both unlabeled fractions, every
seed) it checks that the checkpoint, its log and its sanity check exist, that it was trained on
the L4 (report Sec. 5.3) with the full number of steps, and that it passes the collapse criterion
fixed before the main runs (step E18), against row A with the same width (mean over its seeds):
  collapse        embedding std below twice A's, or k-means NMI not above A's
  failed training k-NN at 100% labels less than KNN_MARGIN above A's
It prints one line per run and the GPU-hours of each row, and exits with an error if any run
is missing or fails a check. It writes nothing.
"""

import argparse
import json
import sys

import numpy as np

from mo433 import config
from mo433.io import banner, data_dir
from mo433.train import main_runs

KNN_MARGIN = 0.10


def check_run(name: str, root, row_a: dict) -> dict:
    """The checks of one main run; ``row_a`` is the sanity of A with the same width."""
    log_path = root / "runs" / f"{name}_log.json"
    sanity_path = root / "sanity" / f"{name}.json"
    if not (root / "runs" / f"{name}.pth").exists() or not log_path.exists() \
            or not sanity_path.exists():
        return {"name": name, "problems": ["missing"]}
    log, sanity = json.loads(log_path.read_text()), json.loads(sanity_path.read_text())
    problems = []
    if "L4" not in (log.get("gpu") or ""):
        problems.append(f"gpu {log.get('gpu')}")
    if len(log["epochs"]) * log["steps_per_epoch"] != log["total_steps"] \
            or len(log["epochs"]) != config.EPOCHS:
        problems.append("steps")
    if sanity["embedding_std"] < 2 * row_a["embedding_std"] \
            or sanity["kmeans_nmi"] <= row_a["kmeans_nmi"]:
        problems.append("collapse")
    elif sanity["knn_100"] < row_a["knn_100"] + KNN_MARGIN:
        problems.append("failed training")
    return {"name": name, "problems": problems, "gpu_hours": log["gpu_hours"],
            "total_steps": log["total_steps"], "skipped_steps": log["skipped_steps"],
            **{k: sanity[k] for k in ("knn_100", "knn_1_mean", "embedding_std", "kmeans_nmi")}}


def check_all(root) -> list:
    row_a = {}
    for c in config.WIDTHS:
        seeds = [json.loads((root / "sanity" / f"A-c{c}-s{s}.json").read_text())
                 for s in config.SEEDS[c]]
        row_a[c] = {k: float(np.mean([a[k] for a in seeds]))
                    for k in ("embedding_std", "kmeans_nmi", "knn_100")}
    return [check_run(name, root, row_a[int(name.split("-")[1][1:])])
            for row in "BC" for name in main_runs(row)]


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    banner("E20: check of the main runs")
    results = check_all(data_dir())
    for r in results:
        if r["problems"] == ["missing"]:
            print(f"  {r['name']:<15} MISSING")
            continue
        print(f"  {r['name']:<15} k-NN {100 * r['knn_100']:5.1f} / {100 * r['knn_1_mean']:5.1f}  "
              f"std {r['embedding_std']:.3f}  NMI {r['kmeans_nmi']:.3f}  "
              f"{r['total_steps']} steps ({r['skipped_steps']} skipped)  "
              f"{60 * r['gpu_hours']:5.1f} min  " + (", ".join(r["problems"]) or "ok"))
    for row in "BC":
        hours = sum(r.get("gpu_hours", 0) for r in results if r["name"][0] == row)
        print(f"  row {row}: {hours:.2f} GPU-h")
    bad = [r["name"] for r in results if r["problems"]]
    if bad:
        sys.exit(f"✗ {len(bad)} of {len(results)} runs fail: {', '.join(bad)}")
    print(f"✓ all {len(results)} main runs present and passing")


if __name__ == "__main__":
    main()
