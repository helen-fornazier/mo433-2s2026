import subprocess

import numpy as np
import torch

from mo433.io import REPO_DIR
from mo433.latency import cpu_list, measure, performance_cpus, summarize


def test_cpu_list_parses_the_sysfs_format():
    assert cpu_list("0-2,5\n") == [0, 1, 2, 5]
    assert cpu_list("7") == [7]


def test_performance_cpus_keeps_one_cpu_per_physical_performance_core(tmp_path):
    (tmp_path / "devices" / "cpu_core").mkdir(parents=True)
    (tmp_path / "devices" / "cpu_core" / "cpus").write_text("0-5\n")
    for siblings in ("0,5", "1-2", "3-4"):
        for c in cpu_list(siblings):
            topo = tmp_path / "devices" / "system" / "cpu" / f"cpu{c}" / "topology"
            topo.mkdir(parents=True)
            (topo / "thread_siblings_list").write_text(siblings)
    assert performance_cpus(tmp_path) == [0, 1, 3]


def test_summarize_and_round_difference():
    s = summarize(np.array([0.1, 0.3]), batch=8)
    assert np.isclose(s["ms_per_batch"], 200) and np.isclose(s["ms_per_batch_std"], 100)
    assert np.isclose(s["ms_per_image"], 25) and np.isclose(s["images_per_second"], 40)
    out = measure({"id": torch.nn.Identity()}, torch.zeros(4, 3, 8, 8), warmup=1, iters=3, rounds=2)
    assert len(out["id"]["rounds"]) == 2 and out["id"]["batches"] == 6
    assert out["id"]["round_diff"] >= 0


def test_makefile_target(tmp_path):
    out = subprocess.run(["make", "-n", "latency", f"D={tmp_path}"], cwd=REPO_DIR,
                         capture_output=True, text=True).stdout
    assert out.strip().endswith("-m mo433.latency")
