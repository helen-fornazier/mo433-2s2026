"""Step E21: choose the SVM hyperparameters once, then freeze them (spec 3.6; DECISOES 20b, 23).

The choice is made on a single encoder, SVM_HPARAM_ENCODER (row B, c16, 100% of the unlabeled
images, seed 0), with 100% of the labels, by stratified SVM_CV_FOLDS-fold cross-validation on
the training split; the test split is never read. The features are preprocessed as in every
evaluation (L2 + StandardScaler fitted on all training features, no labels involved).

  linear SVM   C in SVM_C_GRID, on the whole training split
  RBF SVM      C in SVM_C_GRID x gamma = g / d, g in SVM_G_GRID, on the fixed stratified
               subsample of RBF_MAX_SAMPLES images that the evaluation also uses

Each probe keeps its best grid point (on a tie, the first in grid order: the smaller C, then
the smaller g). A choice on the edge of its grid is flagged, and the grid is then extended in
config. The choice is copied by hand to config.SVM_SELECTED, which the probes (E22) read.

Writes $MO433_DATA/svm_hparams.json: the grid, the CV accuracy of every point (mean and per
fold), the fits that did not converge, and the selection.
"""

import argparse
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from mo433 import config
from mo433.evaluate import fit_preprocessor, linear_svm, rbf_subsample, rbf_svm
from mo433.io import banner, data_dir, load_pickle, save_json, write_meta


def fold_accuracy(clf, x, y, fit, held) -> tuple:
    """Accuracy of ``clf`` fitted on ``fit`` and scored on ``held``; and whether it converged."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(x[fit], y[fit])
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return float((clf.predict(x[held]) == y[held]).mean()), converged


def grid_cv(points: dict, x, y, folds: int = config.SVM_CV_FOLDS, seed: int = 0) -> dict:
    """CV accuracy of each grid point, ``points`` = {key: classifier}; all fits in parallel."""
    splits = list(StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(x, y))
    jobs = [(key, i) for key in points for i in range(folds)]
    out = Parallel(n_jobs=-1)(delayed(fold_accuracy)(points[key], x, y, *splits[i])
                              for key, i in jobs)
    result = {key: {"folds": [], "converged": True} for key in points}
    for (key, _), (acc, converged) in zip(jobs, out):
        result[key]["folds"].append(acc)
        result[key]["converged"] &= converged
    for r in result.values():
        r["mean"] = float(np.mean(r["folds"]))
    return result


def select(result: dict, grids: dict) -> dict:
    """Best point of ``result`` (keys are tuples over ``grids``, in grid order) and the names of
    the hyperparameters whose chosen value is the first or last of its grid."""
    best = max(result, key=lambda k: result[k]["mean"])   # max keeps the first on a tie
    edge = [name for name, v in zip(grids, best) if v in (grids[name][0], grids[name][-1])]
    return {**dict(zip(grids, best)), "cv_mean": result[best]["mean"], "at_edge": edge}


def svm_hparams(feats, y) -> dict:
    """Grid search of both SVM probes on one encoder's training features (labels ``y``)."""
    x = fit_preprocessor(feats).transform(feats)
    dim = x.shape[1]
    sub = rbf_subsample(y)
    grids = {"linear": {"C": config.SVM_C_GRID},
             "rbf": {"C": config.SVM_C_GRID, "g": config.SVM_G_GRID}}
    out = {"feature_dim": dim, "n_linear": len(y), "n_rbf": len(sub), "folds": config.SVM_CV_FOLDS}
    for probe, grid in grids.items():
        start = time.perf_counter()
        if probe == "linear":
            points = {(C,): linear_svm(C) for C in grid["C"]}
            result = grid_cv(points, x, y)
        else:
            points = {(C, g): rbf_svm(C, g, dim) for C in grid["C"] for g in grid["g"]}
            result = grid_cv(points, x[sub], y[sub])
        out[probe] = {"grid": grid, "seconds": round(time.perf_counter() - start, 1),
                      "points": [{**dict(zip(grid, k)), **r} for k, r in result.items()],
                      "selected": select(result, grid)}
    return out


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    root = data_dir()
    out = root / "svm_hparams.json"
    encoder = config.SVM_HPARAM_ENCODER
    banner(f"E21: SVM hyperparameters ({encoder}, {config.SVM_CV_FOLDS}-fold CV on the training split)")
    labels = np.load(root / "eurosat" / "labels.npy")
    y = labels[load_pickle(root / "split_idx.pkl")["train"]]
    feats = np.load(root / "feats" / f"{encoder}_train.npy")

    result = dict(svm_hparams(feats, y), encoder=encoder)
    for probe in ("linear", "rbf"):
        r = result[probe]
        for p in r["points"]:
            print(f"  {probe:<6} " + "  ".join(f"{k} {p[k]:<5g}" for k in r["grid"])
                  + f"  CV {100 * p['mean']:5.2f} (±{100 * np.std(p['folds']):.2f})"
                  + ("" if p["converged"] else "  not converged"))
        s = r["selected"]
        print(f"  -> {probe}: " + ", ".join(f"{k} = {s[k]:g}" for k in r["grid"])
              + f"  ({r['seconds']:.0f} s)"
              + (f"  WARNING: on the edge of the grid ({', '.join(s['at_edge'])})"
                 if s["at_edge"] else ""))

    save_json(result, out)
    write_meta(out, encoder=encoder, folds=config.SVM_CV_FOLDS)
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
