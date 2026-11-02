import json
import subprocess
import sys

import numpy as np

from mo433 import config
from mo433.evaluate import rbf_subsample
from mo433.io import REPO_DIR, save_pickle
from mo433.svm_hparams import main, select


def test_select_keeps_the_first_best_point_and_flags_the_edges():
    grid = {"C": (0.1, 1, 10), "g": (0.3, 1, 3)}
    result = {(C, g): {"mean": 0.5} for C in grid["C"] for g in grid["g"]}
    result[(1, 1)]["mean"] = result[(10, 1)]["mean"] = 0.9
    assert select(result, grid) == {"C": 1, "g": 1, "cv_mean": 0.9, "at_edge": []}
    result[(1, 3)]["mean"] = 0.95
    assert select(result, grid)["at_edge"] == ["g"]


def test_rbf_subsample_is_fixed_stratified_and_only_when_needed():
    y = np.repeat(np.arange(10), [3000, 3000, 3000, 2500, 2500, 2000, 2500, 3000, 2500, 3000])
    sub = rbf_subsample(y)
    assert len(sub) == config.RBF_MAX_SAMPLES and np.all(np.diff(sub) > 0)
    assert np.array_equal(sub, rbf_subsample(y))
    assert np.allclose(np.bincount(y[sub]) / len(sub), np.bincount(y) / len(y), atol=1e-3)
    assert np.array_equal(rbf_subsample(y[:500]), np.arange(500))


def test_main_never_reads_the_test_split(tmp_path, monkeypatch):
    # only the training features exist: any read of the test split would fail
    rng = np.random.default_rng(0)
    n_train, n_test = 300, 100
    y = np.repeat(np.arange(10), (n_train + n_test) // 10)
    (tmp_path / "eurosat").mkdir()
    (tmp_path / "feats").mkdir()
    np.save(tmp_path / "eurosat" / "labels.npy", y)
    train = rng.permutation(len(y))[:n_train]
    save_pickle({"train": train, "test": np.setdiff1d(np.arange(len(y)), train)},
                tmp_path / "split_idx.pkl")
    feats = rng.normal(size=(10, 8))[y[train]] * 2 + rng.normal(size=(n_train, 8))
    np.save(tmp_path / "feats" / f"{config.SVM_HPARAM_ENCODER}_train.npy", feats)
    monkeypatch.setenv("MO433_DATA", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["svm_hparams"])
    main()
    out = json.loads((tmp_path / "svm_hparams.json").read_text())
    assert len(out["linear"]["points"]) == len(config.SVM_C_GRID)
    assert len(out["rbf"]["points"]) == len(config.SVM_C_GRID) * len(config.SVM_G_GRID)
    assert out["rbf"]["selected"]["C"] in config.SVM_C_GRID
    assert out["linear"]["selected"]["cv_mean"] > 0.8 and out["rbf"]["selected"]["cv_mean"] > 0.8


def test_makefile_uses_the_configured_encoder(tmp_path):
    out = subprocess.run(["make", "-n", "svm-hparams", f"D={tmp_path}"], cwd=REPO_DIR,
                         capture_output=True, text=True).stdout
    assert f"mo433.train {config.SVM_HPARAM_ENCODER}\n" in out
    assert out.rstrip().endswith("mo433.svm_hparams")


def test_selected_values_are_on_the_grid():
    assert config.SVM_SELECTED["linear"]["C"] in config.SVM_C_GRID
    assert config.SVM_SELECTED["rbf"]["C"] in config.SVM_C_GRID
    assert config.SVM_SELECTED["rbf"]["g"] in config.SVM_G_GRID
