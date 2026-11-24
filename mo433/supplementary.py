"""Supplementary analyses quoted in the report's discussion and limitations (report Secs. 7, 8).

    python -m mo433.supplementary       (on the CPU machine, after the probes)

1. Row C versus the teacher D, paired on the same test images (report Sec. 7, Q2): the linear
   SVM of each main C run (100% of the unlabeled data) and of D is fitted on the same labeled
   subset, exactly as in the probes; for every (run, fraction, draw) we count the test images
   that only C classifies correctly and those that only D does, and apply the exact McNemar test
   (two-sided binomial test on these discordant pairs). At 100% there is one subset, so one
   test per run.
2. How closely C imitates the teacher: the final distillation loss of every main C run is
   1 - cos, so 1 - loss is the mean cosine between projected student and teacher features.
3. Transient loss spikes during training (report Sec. 8): a spike is an epoch, after the
   warm-up, whose loss is more than SPIKE_RISE above the previous epoch; it is recovered when the
   loss returns below its value before the spike. Also reported: the largest epoch-to-epoch rise
   after the warm-up of every run of each row.

Writes $MO433_DATA/supplementary.json.
"""

import argparse
import json

import numpy as np
from scipy.stats import binomtest

from mo433 import config
from mo433.evaluate import fit_preprocessor, linear_svm
from mo433.io import banner, data_dir, load_pickle, save_json, write_meta
from mo433.train import main_runs

SPIKE_RISE = 0.25
FRACTIONS = (0.01, 1.0)


def linear_predictions(root, encoder: str, y_train, idx_sets: dict) -> dict:
    """Test predictions of the linear SVM fitted on each labeled subset in ``idx_sets``."""
    pre = fit_preprocessor(np.load(root / "feats" / f"{encoder}_train.npy"))
    tr = pre.transform(np.load(root / "feats" / f"{encoder}_train.npy"))
    te = pre.transform(np.load(root / "feats" / f"{encoder}_test.npy"))
    out = {}
    for key, idx in idx_sets.items():
        clf = linear_svm(config.SVM_SELECTED["linear"]["C"]).fit(tr[idx], y_train[idx])
        out[key] = clf.predict(te)
    return out


def mcnemar(pred_c, pred_d, y) -> dict:
    only_c = int(((pred_c == y) & (pred_d != y)).sum())
    only_d = int(((pred_c != y) & (pred_d == y)).sum())
    p = binomtest(only_c, only_c + only_d).pvalue if only_c + only_d else 1.0
    return {"acc_c": float((pred_c == y).mean()), "acc_d": float((pred_d == y).mean()),
            "only_c": only_c, "only_d": only_d, "p": float(p)}


def spikes(losses: list, warmup_epochs: int) -> list:
    """(epoch, rise, epochs to recover) of every spike after the warm-up."""
    out = []
    for e in range(max(warmup_epochs, 1), len(losses)):
        rise = losses[e] / losses[e - 1] - 1
        if rise > SPIKE_RISE and (not out or e > out[-1]["epoch"] - 1 + out[-1]["recovered_in"]):
            back = next((f for f in range(e + 1, len(losses)) if losses[f] <= losses[e - 1]), None)
            out.append({"epoch": e + 1, "rise": round(rise, 3),
                        "recovered_in": None if back is None else back - e})
    return out


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    banner("Supplementary analyses: C vs D (McNemar), imitation, loss spikes")
    root = data_dir()
    labels = np.load(root / "eurosat" / "labels.npy")
    split = load_pickle(root / "split_idx.pkl")
    y_train, y_test = labels[split["train"]], labels[split["test"]]
    subsets = load_pickle(root / "label_subsets.pkl")
    # 100% has a single subset (the whole training split), whatever the draw
    idx_sets = {(f, d): np.asarray(subsets[(f, d)]) for f, d in subsets
                if f in FRACTIONS and (f < 1 or d == 0)}

    pred_d = linear_predictions(root, "D", y_train, idx_sets)
    paired = []
    for run in main_runs("C"):
        if "-d100-" not in run:
            continue
        pred_c = linear_predictions(root, run, y_train, idx_sets)
        for (f, d) in idx_sets:
            paired.append({"run": run, "fraction": f, "draw": d,
                           **mcnemar(pred_c[(f, d)], pred_d[(f, d)], y_test)})
    for r in paired:
        print(f"  {r['run']:<15} {100 * r['fraction']:>4g}% draw {r['draw']}  C {100 * r['acc_c']:5.1f}"
              f"  D {100 * r['acc_d']:5.1f}  only C {r['only_c']:3d}  only D {r['only_d']:3d}"
              f"  p {r['p']:.3f}")

    logs = {run: json.loads((root / "runs" / f"{run}_log.json").read_text())
            for row in "BC" for run in main_runs(row)}
    cosine = {run: 1 - log["epochs"][-1]["loss"] for run, log in logs.items() if run[0] == "C"}
    print(f"  final cosine of C (d100): "
          + ", ".join(f"{c:.3f}" for r, c in cosine.items() if "-d100-" in r))
    loss_spikes, max_rise = {}, {}
    for run, log in logs.items():
        losses = [e["loss"] for e in log["epochs"]]
        warmup_epochs = -(-log["warmup_steps"] // log["steps_per_epoch"])
        loss_spikes[run] = spikes(losses, warmup_epochs)
        max_rise[run] = max(losses[e] / losses[e - 1] - 1 for e in range(warmup_epochs, len(losses)))
    for row in "BC":
        runs = [r for r in logs if r[0] == row]
        n = sum(bool(loss_spikes[r]) for r in runs)
        print(f"  row {row}: {n} of {len(runs)} runs with spikes; largest rise "
              f"{100 * max(max_rise[r] for r in runs):.1f}%")

    out = root / "supplementary.json"
    save_json({"c_vs_d": paired, "final_cosine": cosine, "spike_rise": SPIKE_RISE,
               "spikes": loss_spikes, "max_rise": max_rise}, out)
    write_meta(out, svm=config.SVM_SELECTED, spike_rise=SPIKE_RISE)
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
