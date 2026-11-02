import re
import subprocess

import numpy as np

from mo433 import config
from mo433.io import REPO_DIR
from mo433.train import parse_run
from mo433.tuning import knn_cv_accuracy, trial_names


def test_trial_names_encode_the_grid():
    for row in "BC":
        grid = {(parse_run(n)["lr"], parse_run(n)["hidden"]) for n in trial_names(row)}
        assert grid == {(lr, h) for lr in config.TUNING_LRS for h in config.TUNING_HIDDEN}
        assert all(parse_run(n)["epochs"] == config.TUNING_EPOCHS for n in trial_names(row))


def test_makefile_trains_exactly_these_trials(tmp_path):
    out = subprocess.run(["make", "-n", "tuning", f"D={tmp_path}"], cwd=REPO_DIR,
                         capture_output=True, text=True).stdout
    trained = set(re.findall(r"mo433\.train (\S+)", out))
    assert trained == set(trial_names("B") + trial_names("C"))


def test_knn_cv_separates_good_from_random_features():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(10), 50)
    good = rng.normal(size=(10, 16))[y] * 3 + rng.normal(size=(500, 16))
    noise = rng.normal(size=(500, 16))
    assert len(knn_cv_accuracy(good, y)) == config.TUNING_CV_FOLDS
    assert np.mean(knn_cv_accuracy(good, y)) > 0.9 > 0.3 > np.mean(knn_cv_accuracy(noise, y))
