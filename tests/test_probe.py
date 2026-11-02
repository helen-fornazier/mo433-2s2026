import functools
import re
import subprocess

import numpy as np

from mo433 import config, probe
from mo433.evaluate import rbf_subsample
from mo433.io import REPO_DIR
from mo433.label_subsets import make_label_subsets
from mo433.probe import PROBES, probe_encoder, subsets_to_fit
from mo433.train import main_runs


def data(seed=0, n_train=2000, n_test=500):
    rng = np.random.default_rng(seed)
    y_tr, y_te = rng.permutation(np.arange(n_train) % 10), rng.permutation(np.arange(n_test) % 10)
    centers = rng.normal(size=(10, 16))
    tr = centers[y_tr] + 0.3 * rng.normal(size=(n_train, 16))
    te = centers[y_te] + 0.3 * rng.normal(size=(n_test, 16))
    return tr, y_tr, te, y_te, make_label_subsets(y_tr)


def test_100_percent_is_fitted_once_and_other_fractions_per_draw():
    keys = subsets_to_fit({(f, d): None for f in config.LABEL_FRACTIONS for d in config.LABEL_DRAWS})
    assert keys == [(f, d) for f in config.LABEL_FRACTIONS if f < 1 for d in config.LABEL_DRAWS] \
        + [(1.0, 0)]


def test_separable_features_score_high_and_shuffled_labels_score_chance():
    tr, y_tr, te, y_te, subsets = data()
    rows = probe_encoder(tr, y_tr, te, y_te, subsets, n_jobs=1)
    assert len(rows) == len(subsets_to_fit(subsets)) * len(PROBES)
    assert all(r["accuracy"] > 0.95 for r in rows if r["fraction"] >= 0.1)
    shuffled = np.random.default_rng(1).permutation(y_tr)
    rows = probe_encoder(tr, shuffled, te, y_te, make_label_subsets(shuffled), n_jobs=1)
    assert all(r["accuracy"] < 0.2 for r in rows if r["fraction"] == 1.0)


def test_rbf_is_fitted_on_the_fixed_subsample_above_the_limit(monkeypatch):
    monkeypatch.setattr(probe, "rbf_subsample", functools.partial(rbf_subsample, max_samples=500))
    tr, y_tr, te, y_te, subsets = data()
    rows = probe_encoder(tr, y_tr, te, y_te, subsets, n_jobs=1)
    n_fit = {(r["fraction"], r["probe"]): r["n_fit"] for r in rows}
    assert n_fit[(1.0, "rbf")] == n_fit[(0.5, "rbf")] == 500
    assert n_fit[(1.0, "linear")] == 2000 and n_fit[(0.1, "rbf")] == 200


def test_makefile_probes_the_26_encoders(tmp_path):
    out = subprocess.run(["make", "-n", "probes", f"D={tmp_path}"], cwd=REPO_DIR,
                         capture_output=True, text=True).stdout
    probed = re.findall(r"mo433\.probe (\S+)", out)
    expected = ([f"A-c{c}-s{s}" for c in config.WIDTHS for s in config.SEEDS[c]]
                + main_runs("B") + main_runs("C") + ["D"])
    assert sorted(probed) == sorted(expected) and len(probed) == 26
