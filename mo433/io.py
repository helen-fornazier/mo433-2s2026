"""Paths, atomic writes and run metadata.

All data and outputs live under ``$MO433_DATA``:
  CPU machine ("laptop"): ../data relative to the repository (default when it is unset)
  Colab:  /content/drive/MyDrive/mo433-2s2026

Every output is written atomically (to ``<name>.tmp``, then renamed), so an interrupted run
never leaves a file that make would take as finished. Next to each output goes a
``<name>.json`` with the configuration and the environment that produced it.
"""

import json
import os
import pickle
import platform
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
TRACKED_PACKAGES = ("torch", "torchvision", "timm", "lightly", "scikit-learn",
                    "numpy", "scipy", "pandas", "matplotlib")


def data_dir() -> Path:
    """Root of all data and outputs ($MO433_DATA, default ../data next to the repo)."""
    return Path(os.environ.get("MO433_DATA", REPO_DIR.parent / "data")).resolve()


def host_label() -> str:
    """'colab' on Google Colab, 'laptop' elsewhere."""
    import importlib.util
    try:
        has_colab_module = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:                  # no "google" package at all
        has_colab_module = False
    return "colab" if "COLAB_RELEASE_TAG" in os.environ or has_colab_module else "laptop"


@contextmanager
def atomic_path(path):
    """Yield a temporary path; rename it to ``path`` only if the block succeeds."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def save_pickle(obj, path):
    with atomic_path(path) as tmp, open(tmp, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    with atomic_path(path) as tmp, open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=str)
        f.write("\n")


def git_state() -> dict:
    """Commit hash of the repository and whether the working tree had local changes."""
    def git(*args):
        return subprocess.run(["git", "-C", str(REPO_DIR), *args], capture_output=True,
                              text=True).stdout.strip()
    try:
        return {"commit": git("rev-parse", "HEAD") or None,
                "dirty": bool(git("status", "--porcelain"))}
    except FileNotFoundError:
        return {"commit": None, "dirty": None}


def package_versions() -> dict:
    versions = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def device_info() -> dict:
    """CPU and (if present) GPU description; torch is imported lazily."""
    info = {"cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    try:
        import torch
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except ImportError:
        info["cuda"] = None
    return info


def environment() -> dict:
    return {"host": host_label(), "hostname": socket.gethostname(),
            "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": package_versions(), "device": device_info(), "git": git_state()}


def write_meta(output_path, **info):
    """Write ``<output>.json`` with the given info plus environment, command and timestamp."""
    output_path = Path(output_path)
    meta = {"output": output_path.name, "command": " ".join(sys.argv),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S %z"), **info,
            "environment": environment()}
    save_json(meta, output_path.with_name(output_path.name + ".json"))
    return meta


def banner(title: str):
    print("=" * 60)
    print(title)
    print("=" * 60)
