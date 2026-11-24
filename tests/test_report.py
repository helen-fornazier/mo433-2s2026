import json
import subprocess

import numpy as np
import pandas as pd

from mo433 import config
from mo433.io import REPO_DIR
from mo433.report import aggregate, encoders, gap, group_of, load_probes, parse, pm, table_main


def synthetic(tmp_path, names, acc):
    """probes/<name>.json with accuracy acc(name, fraction, draw, probe), 100% fitted once."""
    (tmp_path / "probes").mkdir()
    for name in names:
        rows = [{"fraction": f, "draw": d, "probe": p, "accuracy": acc(name, f, d, p)}
                for f in config.LABEL_FRACTIONS for d in (config.LABEL_DRAWS if f < 1 else (0,))
                for p in ("knn", "linear", "rbf")]
        (tmp_path / "probes" / f"{name}.json").write_text(json.dumps({"results": rows}))


def test_names_and_groups():
    assert len(encoders()) == 26
    assert group_of("B-c16-d50-s2") == "B-c16-d50" and group_of("A-c32-s1") == "A-c32"
    assert group_of("D") == "D" and parse("C-c32-d100-s1") == {"row": "C", "c": 32, "d": 100, "seed": 1}


def test_aggregation_over_seeds_and_draws(tmp_path):
    names = ["B-c16-d100-s0", "B-c16-d100-s1", "B-c16-d100-s2", "D"]
    seed_vals = {"s0": 50.0, "s1": 50.0, "s2": 60.0}
    synthetic(tmp_path, names, lambda n, f, d, p: 0.9 + 0.01 * d if n == "D"
              else seed_vals[n[-2:]] / 100 + 0.001 * d)
    agg = aggregate(load_probes(tmp_path, names))
    # below 100%: 3 seeds x 3 draws = 9 values, sample s.d.
    v = np.array([seed_vals[s] + 0.1 * d for s in seed_vals for d in range(3)])
    r = agg.loc[("B-c16-d100", 0.1, "linear")]
    assert r["n"] == 9 and np.isclose(r["mean"], v.mean()) and np.isclose(r["sd"], v.std(ddof=1))
    # at 100%: seeds only
    r = agg.loc[("B-c16-d100", 1.0, "knn")]
    assert r["n"] == 3 and np.isclose(r["sd"], np.std([50, 50, 60], ddof=1))
    # D: s.d. over draws below 100%, a single value at 100%
    assert np.isclose(agg.loc[("D", 0.01, "rbf"), "sd"], np.std([90, 91, 92], ddof=1))
    assert agg.loc[("D", 1.0, "rbf"), "n"] == 1 and np.isnan(agg.loc[("D", 1.0, "rbf"), "sd"])


def test_gap_and_the_beyond_the_spread_criterion():
    idx = pd.MultiIndex.from_tuples([("B-c16-d100", 0.01, "linear"), ("C-c16-d100", 0.01, "linear")],
                                    names=["group", "fraction", "probe"])
    agg = pd.DataFrame({"mean": [80.0, 83.0], "sd": [3.0, 4.0], "n": [9, 9]}, index=idx)
    g = gap(agg, 16, 100, 0.01, "linear")
    assert g == {"gap": 3.0, "spread": 5.0, "beyond": False}
    agg.loc[("C-c16-d100", 0.01, "linear"), "mean"] = 85.5
    assert gap(agg, 16, 100, 0.01, "linear")["beyond"]


def test_format_and_bold_of_the_best_student(tmp_path):
    assert pm(91.04, 0.55) == "91.0\\,$\\pm$\\,0.6" and pm(97.2, np.nan) == "97.2"
    names = encoders()
    row_acc = {"A": 0.5, "B": 0.8, "C": 0.9, "D": 0.95}
    synthetic(tmp_path, names, lambda n, f, d, p: row_acc[n[0]] + 0.001 * d)
    tex = table_main(aggregate(load_probes(tmp_path, names)))
    bold = [line for line in tex.splitlines() if "textbf" in line]
    assert len(bold) == 3 * 2 and all("rowC" in line for line in bold)    # never D, A or B
    assert tex.count("\\rowD") == 3


def test_makefile_target(tmp_path):
    out = subprocess.run(["make", "-n", "report", f"D={tmp_path}", f"REPORT={tmp_path}/r"],
                         cwd=REPO_DIR, capture_output=True, text=True).stdout
    assert out.strip().endswith("-m mo433.report")
