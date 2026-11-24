"""Step E17: select (lr, h) for rows B and C from the tuning trials (spec 3.6; report Sec. 5.2).

Each row gets the same grid, TUNING_LRS x TUNING_HIDDEN, trained for TUNING_EPOCHS epochs with
c=TUNING_WIDTH and seed TUNING_SEED (runs <row>-c16-d100-s0-tune-lr<lr>-h<h>). The criterion is
the same for both rows and never touches the test split: k-NN accuracy (the probe of spec 3.6)
by stratified TUNING_CV_FOLDS-fold cross-validation on the training split's features. As in
every evaluation, the features are L2-normalized and standardized with a scaler fitted on all
training features (no labels involved). Each row keeps its own best trial; the choice is copied
by hand to config.SELECTED, which the main runs read.

Writes $MO433_DATA/tuning.json: every trial's CV accuracy (mean and per fold), its final loss,
GPU-hours and embedding std (a collapsed trial would show it), and the selection per row. A trial
whose training diverged (mo433.train) is recorded with the epoch where it did, has no accuracy,
and cannot be selected.
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold

from mo433 import config
from mo433.evaluate import embedding_std, fit_preprocessor, knn_accuracy
from mo433.io import banner, data_dir, load_pickle, save_json, write_meta


def trial_name(row: str, lr: float, hidden: int) -> str:
    return (f"{row}-c{config.TUNING_WIDTH}-d100-s{config.TUNING_SEED}"
            f"-tune-lr{lr:.0e}-h{hidden}").replace("e-0", "e-")


def trial_names(row: str) -> list:
    return [trial_name(row, lr, h) for lr in config.TUNING_LRS for h in config.TUNING_HIDDEN]


def knn_cv_accuracy(feats, y, folds: int = config.TUNING_CV_FOLDS,
                    seed: int = config.TUNING_CV_SEED) -> list:
    """k-NN accuracy of each fold of a stratified K-fold split of the training split."""
    x = fit_preprocessor(feats).transform(feats)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [knn_accuracy(x[fit], y[fit], x[held], y[held])["accuracy"]
            for fit, held in cv.split(x, y)]


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    root = data_dir()
    out = root / "tuning.json"
    banner("E17: tuning selection (k-NN, 5-fold CV on the training split)")
    labels = np.load(root / "eurosat" / "labels.npy")
    y = labels[load_pickle(root / "split_idx.pkl")["train"]]

    result = {"criterion": f"knn_cv{config.TUNING_CV_FOLDS}", "trials": {}, "selected": {}}
    for row in "BC":
        for name in trial_names(row):
            log = json.loads((root / "runs" / f"{name}_log.json").read_text())
            if "diverged_at_epoch" in log:
                result["trials"][name] = {
                    "row": row, "lr": log["lr"], "hidden": log["hidden"], "knn_cv_mean": None,
                    "diverged_at_epoch": log["diverged_at_epoch"], "gpu_hours": log["gpu_hours"]}
                print(f"  {name:<36} diverged at epoch {log['diverged_at_epoch']}")
                continue
            feats = np.load(root / "feats" / f"{name}_train.npy")
            folds = knn_cv_accuracy(feats, y)
            result["trials"][name] = {
                "row": row, "lr": log["lr"], "hidden": log["hidden"],
                "knn_cv_mean": float(np.mean(folds)), "knn_cv_folds": folds,
                "final_loss": log["epochs"][-1]["loss"], "gpu_hours": log["gpu_hours"],
                "embedding_std": embedding_std(feats)}
            r = result["trials"][name]
            print(f"  {name:<36} k-NN CV {100 * r['knn_cv_mean']:5.2f}  "
                  f"(±{100 * np.std(folds):.2f})  loss {r['final_loss']:.4f}  "
                  f"std {r['embedding_std']:.3f}")
        best = max((n for n in trial_names(row) if result["trials"][n]["knn_cv_mean"] is not None),
                   key=lambda n: result["trials"][n]["knn_cv_mean"])
        t = result["trials"][best]
        result["selected"][row] = {"trial": best, "lr": t["lr"], "hidden": t["hidden"],
                                   "knn_cv_mean": t["knn_cv_mean"]}
        print(f"  -> row {row}: {best}")

    save_json(result, out)
    write_meta(out, folds=config.TUNING_CV_FOLDS, seed=config.TUNING_CV_SEED)
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
