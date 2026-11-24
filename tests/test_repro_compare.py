import json
import subprocess

import pytest

from mo433 import config
from mo433.io import REPO_DIR
from mo433.repro_compare import compare


def write_probes(root, names, acc):
    (root / "probes").mkdir(parents=True)
    for name in names:
        rows = [{"fraction": f, "draw": d, "probe": p, "accuracy": acc(name, d)}
                for f in config.LABEL_FRACTIONS for d in (config.LABEL_DRAWS if f < 1 else (0,))
                for p in ("knn", "linear", "rbf")]
        (root / "probes" / f"{name}.json").write_text(json.dumps({"results": rows}))


def test_compare_uses_the_common_encoders_and_measures_the_difference(tmp_path):
    ref, new = tmp_path / "ref", tmp_path / "new"
    write_probes(ref, ["B-c16-d100-s0", "B-c16-d100-s1", "D"], lambda n, d: 0.90 + 0.01 * d)
    write_probes(new, ["B-c16-d100-s0", "D"], lambda n, d: 0.905 + 0.01 * d)   # partial
    r = compare(ref, new)
    assert r["encoders"] == ["B-c16-d100-s0", "D"]
    assert r["n_cells"] == 2 * 4 * 3                          # 2 groups x 4 fractions x 3 probes
    assert r["max_abs_diff"] == pytest.approx(0.5) and r["mean_abs_diff"] == pytest.approx(0.5)
    # the original s.d. over draws is 1 point below 100% and undefined (one value) at 100%
    assert r["cells_within_sd"] == 2 * 3 * 3


def test_makefile_groups_cover_the_pipeline(tmp_path):
    out = subprocess.run(["make", "-n", "all", f"D={tmp_path}", f"REPORT={tmp_path}/r"],
                         cwd=REPO_DIR, capture_output=True, text=True).stdout
    steps = {"dataset", "views", "teacher_cache --reference", "teacher_cache\n", "train B-c32-d50-s1",
             "train C-c16-d100-s2", "svm_hparams", "probe D", "diagnostics D", "latency", "report"}
    assert all(f"mo433.{s}" in out for s in steps)
    assert "tune-lr" not in out and "lr1e-2" not in out and "mo433.throughput" not in out
