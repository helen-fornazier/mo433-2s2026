import re
import subprocess

import numpy as np

from mo433.diagnostics import TSNE_PARAMS, tsne_projection
from mo433.io import REPO_DIR


def test_tsne_is_deterministic_and_keeps_separated_classes_apart():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 60)
    x = 10 * rng.normal(size=(3, 8))[y] + rng.normal(size=(len(y), 8))
    proj = tsne_projection(x)
    assert proj.shape == (len(y), 2) and proj.dtype == np.float32
    assert np.array_equal(proj, tsne_projection(x))
    centers = np.stack([proj[y == c].mean(0) for c in range(3)])
    spread = max(np.linalg.norm(proj[y == c] - centers[c], axis=1).mean() for c in range(3))
    assert min(np.linalg.norm(centers[i] - centers[j]) for i in range(3) for j in range(i)) > 2 * spread
    assert TSNE_PARAMS["init"] == "pca" and TSNE_PARAMS["perplexity"] == 30


def test_makefile_runs_the_diagnostics_of_the_26_encoders(tmp_path):
    out = subprocess.run(["make", "-n", "diagnostics", f"D={tmp_path}"], cwd=REPO_DIR,
                         capture_output=True, text=True).stdout
    assert len(set(re.findall(r"mo433\.diagnostics (\S+)", out))) == 26
