"""Step E22: the probes of one encoder on its cached features (spec 3.6; DECISOES 20-24).

    python -m mo433.probe <encoder>        e.g. A-c16-s0, B-c16-d100-s0, C-c32-d50-s1, D

For every labeled subset of label_subsets.pkl (fractions 1, 10, 50 and 100%, three draws each)
it fits the three probes on the subset's training features and scores them on the whole test
split:
  knn      cosine k-NN, k = min(20, smallest class in the subset)
  linear   LinearSVC with the frozen C of config.SVM_SELECTED
  rbf      SVC(kernel="rbf") with the frozen (C, g), on the subset itself or, above
           RBF_MAX_SAMPLES images, on its fixed stratified subsample
The preprocessing is the one of every evaluation (L2 + StandardScaler fitted on all training
features of the encoder). The encoder is never run here: only its cached features are read.

At 100% the three draws are the same set (the whole training split), and every probe is then
deterministic (the RBF subsample depends only on the labels), so 100% is fitted once, as draw 0.

Writes $MO433_DATA/probes/<encoder>.json: one row per fraction x draw x probe, with the test
accuracy, the number of fitting examples and the fit time.
"""

import argparse
import time
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning

from mo433 import config
from mo433.evaluate import fit_preprocessor, knn_accuracy, linear_svm, rbf_subsample, rbf_svm
from mo433.io import banner, data_dir, load_pickle, save_json, write_meta

PROBES = ("knn", "linear", "rbf")


def subsets_to_fit(label_subsets: dict) -> list:
    """(fraction, draw) of the subsets to fit: every draw below 100%, one draw at 100%."""
    return sorted(k for k in label_subsets if k[0] < 1.0 or k[1] == min(config.LABEL_DRAWS))


def fit_probe(probe: str, tr, y_tr, te, y_te, idx) -> dict:
    """Test accuracy of one probe fitted on the training rows ``idx``."""
    start = time.perf_counter()
    out = {}
    if probe == "knn":
        r = knn_accuracy(tr[idx], y_tr[idx], te, y_te)
        out.update(accuracy=r["accuracy"], k=r["k"], n_fit=len(idx))
    else:
        if probe == "linear":
            clf = linear_svm(config.SVM_SELECTED["linear"]["C"])
        else:
            idx = idx[rbf_subsample(y_tr[idx])]
            clf = rbf_svm(config.SVM_SELECTED["rbf"]["C"], config.SVM_SELECTED["rbf"]["g"],
                          tr.shape[1])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(tr[idx], y_tr[idx])
        out.update(accuracy=float((clf.predict(te) == y_te).mean()), n_fit=len(idx),
                   converged=not any(issubclass(w.category, ConvergenceWarning) for w in caught))
    return dict(out, seconds=round(time.perf_counter() - start, 2))


def probe_encoder(train_feats, y_train, test_feats, y_test, label_subsets: dict,
                  n_jobs: int = -1) -> list:
    """One row per (fraction, draw, probe); the fits run in parallel."""
    pre = fit_preprocessor(train_feats)
    tr, te = pre.transform(train_feats), pre.transform(test_feats)
    jobs = [(frac, draw, probe) for frac, draw in subsets_to_fit(label_subsets)
            for probe in PROBES]
    results = Parallel(n_jobs=n_jobs)(
        delayed(fit_probe)(probe, tr, y_train, te, y_test, np.asarray(label_subsets[(frac, draw)]))
        for frac, draw, probe in jobs)
    return [{"fraction": frac, "draw": draw, "probe": probe, **r}
            for (frac, draw, probe), r in zip(jobs, results)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoder", help="an encoder with cached features")
    args = parser.parse_args()
    root = data_dir()
    out = root / "probes" / f"{args.encoder}.json"
    banner(f"E22: probes of {args.encoder}")
    labels = np.load(root / "eurosat" / "labels.npy")
    split = load_pickle(root / "split_idx.pkl")
    train_feats = np.load(root / "feats" / f"{args.encoder}_train.npy")
    test_feats = np.load(root / "feats" / f"{args.encoder}_test.npy")

    start = time.perf_counter()
    rows = probe_encoder(train_feats, labels[split["train"]], test_feats, labels[split["test"]],
                         load_pickle(root / "label_subsets.pkl"))
    seconds = time.perf_counter() - start
    for frac in sorted({r["fraction"] for r in rows}):
        line = "  ".join(
            f"{p} {100 * np.mean([r['accuracy'] for r in rows if r['fraction'] == frac and r['probe'] == p]):5.1f}"
            for p in PROBES)
        print(f"  {100 * frac:>5g}%  {line}")
    unconverged = [r for r in rows if r.get("converged") is False]
    if unconverged:
        print(f"  WARNING: {len(unconverged)} SVM fits did not converge")
    save_json({"encoder": args.encoder, "feature_dim": int(train_feats.shape[1]),
               "svm": config.SVM_SELECTED, "seconds": round(seconds, 1), "results": rows}, out)
    write_meta(out, encoder=args.encoder, svm=config.SVM_SELECTED)
    print(f"✓ {args.encoder}: {seconds:.0f} s, written to {out}")


if __name__ == "__main__":
    main()
