"""Step E13: the teacher's features on every pre-generated view, the targets of row C (spec 3.5, 3.9).

Two outputs, written in this order:

  teacher_ref.npz        (``--reference``, on the laptop's CPU) fp32 teacher features of
                         TEACHER_REF_VIEWS views drawn with a fixed seed, plus their positions.
  teacher_feats_K10.npy  (on Colab's GPU) shape (N_train, K, 384), float16: row i, column k is
                         the teacher's feature of views_K10.npy[i, k]. Computed with the same
                         ``extract`` and the same fp16 autocast as row D.

Before the cache is written, these checks must pass (otherwise nothing is written):
  - no NaN or Inf after the cast to fp16;
  - the teacher's weights are the ones hashed on the laptop (teacher_info.json);
  - TEACHER_FP32_VIEWS views recomputed in fp32 on the GPU: cosine with the cache above
    TEACHER_CACHE_MIN_COS (fp16 is precise enough);
  - the reference views recomputed in fp32 on the GPU match the laptop's fp32 features (cosine
    above TEACHER_HOSTS_MIN_COS), and so does the cache at the same positions: this is what proves
    that view (i, k) and feature (i, k) line up. The cosine with the *next* view of the same image
    is also recorded, to show that the check would catch a shift.

On Colab the views are first copied from Drive to the local disk (reading 2.65 GB through the
Drive mount, batch by batch, is slow). The wall time and the GPU name go to the .json metadata
(GPU-hours for tab:cost).
"""

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from mo433 import config
from mo433.features import extract
from mo433.io import atomic_path, banner, data_dir, host_label, write_meta
from mo433.teacher import load_teacher, weights_sha256

COLAB_STAGE_DIR = Path("/content/stage")


def sample_views(n_images: int, k_views: int, n: int, seed: int) -> np.ndarray:
    """``n`` distinct (image, view) positions, as an (n, 2) array, sorted."""
    flat = np.random.default_rng(seed).choice(n_images * k_views, size=n, replace=False)
    return np.stack(np.divmod(np.sort(flat), k_views), axis=1)


def gather(views, positions) -> np.ndarray:
    return np.stack([views[i, k] for i, k in positions])


def cosine(a, b) -> np.ndarray:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def stage(path: Path, stage_dir: Path | None) -> Path:
    """Copy ``path`` to ``stage_dir`` (unless an identical-size copy is already there)."""
    if stage_dir is None:
        return path
    local = stage_dir / path.name
    if not local.exists() or local.stat().st_size != path.stat().st_size:
        start = time.perf_counter()
        with atomic_path(local) as tmp:
            shutil.copyfile(path, tmp)
        print(f"  copied {path.name} to {stage_dir} in {time.perf_counter() - start:.0f} s")
    return local


def write_reference(root: Path, views):
    out = root / "teacher_ref.npz"
    positions = sample_views(len(views), views.shape[1], config.TEACHER_REF_VIEWS,
                             config.TEACHER_REF_SEED)
    teacher = load_teacher("cpu", amp=False)
    feats = extract(teacher, gather(views, positions), batch_size=len(positions))
    with atomic_path(out) as tmp, open(tmp, "wb") as f:
        np.savez(f, positions=positions, feats=feats)
    write_meta(out, n_views=len(positions), seed=config.TEACHER_REF_SEED, precision="fp32",
               weights_sha256=weights_sha256(teacher.vit))
    print(f"✓ {len(positions)} reference views written to {out}")


def check_cache(cache, views, device, root) -> dict:
    """The checks of the module docstring; returns the numbers, raises if one fails."""
    import json
    fp32 = load_teacher(device, amp=False)
    expected_hash = json.loads((root / "teacher_info.json").read_text())["weights_sha256"]

    pos = sample_views(len(views), views.shape[1], config.TEACHER_FP32_VIEWS,
                       config.TEACHER_FP32_SEED)
    cos_fp16 = cosine(cache[pos[:, 0], pos[:, 1]], extract(fp32, gather(views, pos)))

    ref = np.load(root / "teacher_ref.npz")
    rpos, rfeats = ref["positions"], ref["feats"]
    cos_hosts = cosine(extract(fp32, gather(views, rpos)), rfeats)
    cos_cache_ref = cosine(cache[rpos[:, 0], rpos[:, 1]], rfeats)
    shifted = (rpos[:, 1] + 1) % views.shape[1]
    cos_next_view = cosine(cache[rpos[:, 0], shifted], rfeats)

    checks = {
        "finite": bool(np.isfinite(cache).all()),
        "weights_match_laptop": weights_sha256(fp32.vit) == expected_hash,
        "fp16_vs_fp32_min_cos": float(cos_fp16.min()),
        "fp16_vs_fp32_mean_cos": float(cos_fp16.mean()),
        "gpu_vs_laptop_fp32_min_cos": float(cos_hosts.min()),
        "cache_vs_laptop_min_cos": float(cos_cache_ref.min()),
        "next_view_vs_laptop_max_cos": float(cos_next_view.max()),
    }
    failed = [name for name, ok in (
        ("finite", checks["finite"]),
        ("weights_match_laptop", checks["weights_match_laptop"]),
        ("fp16_vs_fp32", cos_fp16.min() > config.TEACHER_CACHE_MIN_COS),
        ("gpu_vs_laptop", cos_hosts.min() > config.TEACHER_HOSTS_MIN_COS),
        ("cache_vs_laptop", cos_cache_ref.min() > config.TEACHER_CACHE_MIN_COS)) if not ok]
    for k, v in checks.items():
        print(f"  {k:<30} {v}")
    if failed:
        raise SystemExit(f"teacher cache checks failed: {failed}; nothing written")
    return checks


def write_cache(root: Path, views, device, batch_size: int):
    out = root / f"teacher_feats_K{config.K_VIEWS}.npy"
    n, k = views.shape[:2]
    teacher = load_teacher(device)
    start = time.perf_counter()
    feats = extract(teacher, views.reshape(n * k, *views.shape[2:]), batch_size=batch_size,
                    device=device)
    seconds = time.perf_counter() - start
    cache = feats.astype(np.float16).reshape(n, k, config.TEACHER_DIM)
    print(f"  {n * k} views in {seconds:.0f} s ({n * k / seconds:.0f} views/s)")

    checks = check_cache(cache, views, device, root)
    with atomic_path(out) as tmp, open(tmp, "wb") as f:
        np.save(f, cache)
    write_meta(out, shape=list(cache.shape), dtype="float16", device=device,
               batch_size=batch_size, amp=device == "cuda", seconds=round(seconds, 1),
               gpu_hours=round(seconds / 3600, 4), views_per_second=round(n * k / seconds, 1),
               weights_sha256=weights_sha256(teacher.vit), checks=checks)
    print(f"✓ written to {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference", action="store_true",
                        help="write teacher_ref.npz (fp32, CPU; run on the laptop)")
    parser.add_argument("--batch-size", type=int, default=config.TEACHER_CACHE_BATCH)
    parser.add_argument("--allow-cpu", action="store_true",
                        help="compute the full cache without a GPU (hours)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = data_dir()

    views_path = root / f"views_K{config.K_VIEWS}.npy"
    if args.reference:
        banner("E13: teacher reference features (fp32, CPU)")
        write_reference(root, np.load(views_path, mmap_mode="r"))
        return

    banner(f"E13: teacher cache on all views ({args.device})")
    if args.device == "cpu" and not args.allow_cpu:
        raise SystemExit("no GPU: the full cache runs on Colab (or pass --allow-cpu)")
    stage_dir = COLAB_STAGE_DIR if host_label() == "colab" else None
    views = np.load(stage(views_path, stage_dir), mmap_mode="r")
    write_cache(root, views, args.device, args.batch_size)


if __name__ == "__main__":
    main()
