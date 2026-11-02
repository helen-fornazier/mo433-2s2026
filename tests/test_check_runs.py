import json

from mo433 import config
from mo433.check_runs import check_all, check_run

ROW_A = {"embedding_std": 0.009, "kmeans_nmi": 0.28, "knn_100": 0.56, "knn_1_mean": 0.36}


def write_run(root, name, gpu="NVIDIA L4", epochs=config.EPOCHS, **sanity):
    (root / "runs").mkdir(exist_ok=True)
    (root / "sanity").mkdir(exist_ok=True)
    (root / "runs" / f"{name}.pth").write_bytes(b"")
    log = {"gpu": gpu, "steps_per_epoch": 42, "total_steps": 42 * config.EPOCHS,
           "skipped_steps": 0, "gpu_hours": 0.25, "epochs": [{}] * epochs}
    (root / "runs" / f"{name}_log.json").write_text(json.dumps(log))
    healthy = {"embedding_std": 0.05, "kmeans_nmi": 0.7, "knn_100": 0.93, "knn_1_mean": 0.86}
    (root / "sanity" / f"{name}.json").write_text(json.dumps({**healthy, **sanity}))


def test_check_run_applies_the_e18_criterion(tmp_path):
    write_run(tmp_path, "ok")
    write_run(tmp_path, "flat", embedding_std=0.01)
    write_run(tmp_path, "nmi", kmeans_nmi=0.28)
    write_run(tmp_path, "weak", knn_100=0.60)
    write_run(tmp_path, "t4", gpu="Tesla T4", epochs=150)
    problems = {n: check_run(n, tmp_path, ROW_A)["problems"]
                for n in ("ok", "flat", "nmi", "weak", "t4", "absent")}
    assert problems == {"ok": [], "flat": ["collapse"], "nmi": ["collapse"],
                        "weak": ["failed training"], "t4": ["gpu Tesla T4", "steps"],
                        "absent": ["missing"]}


def test_check_all_lists_the_20_main_runs(tmp_path):
    (tmp_path / "sanity").mkdir()
    for c in config.WIDTHS:
        for s in config.SEEDS[c]:
            (tmp_path / "sanity" / f"A-c{c}-s{s}.json").write_text(json.dumps(ROW_A))
    results = check_all(tmp_path)
    assert len(results) == 20 and all(r["problems"] == ["missing"] for r in results)
