"""Step E6: pre-generate K augmented views of every training image ($MO433_DATA/views_K10.npy).

Shape (N_train, K, 64, 64, 3), uint8, unnormalized; row i holds the views of the i-th image of
the training split (split_idx.pkl["train"]), the same positions used by unlab_idx_*.pkl.
View k of image i is ``generate_view(image, view_seed(i, k))`` (mo433.transforms), so any view
can be regenerated on its own. B and C both draw from these views (spec 3.9), and the teacher's
features are computed on them once (step E13).

Runs on CPU with one process per core; each worker writes its chunk straight into a memory-mapped
.npy, which is renamed into place only when every chunk is done.
"""

import argparse
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from mo433 import config
from mo433.io import atomic_path, banner, data_dir, load_pickle, write_meta

_IMAGES = None  # training images, set in each worker by _init_worker


def _init_worker(images_path, train_idx, positions):
    global _IMAGES
    import torch
    torch.set_num_threads(1)                     # one core per worker, no oversubscription
    _IMAGES = np.load(images_path, mmap_mode="r")[train_idx[positions]]


def _generate_chunk(args):
    """Generate the K views of ``chunk`` (indices into the selected positions) into ``out_path``."""
    out_path, chunk, positions, k_views = args
    from mo433.transforms import generate_view, make_view_transform, view_seed
    transform = make_view_transform()
    out = np.load(out_path, mmap_mode="r+")
    for j in chunk:
        image = _IMAGES[j]
        for k in range(k_views):
            out[j, k] = generate_view(image, view_seed(int(positions[j]), k), transform)
    out.flush()
    return len(chunk)


def generate_views(images_path, train_idx, positions, out_path, k_views=config.K_VIEWS,
                   workers=None):
    """Write the views of the given training positions to ``out_path`` (a .npy file)."""
    workers = workers or os.cpu_count()
    n = len(positions)
    np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint8,
                              shape=(n, k_views, config.IMAGE_SIZE, config.IMAGE_SIZE, 3)).flush()
    chunks = np.array_split(np.arange(n), max(1, min(n, workers * 8)))
    tasks = [(str(out_path), c, positions, k_views) for c in chunks if len(c)]
    done, next_report = 0, 0.0
    with get_context("fork").Pool(workers, initializer=_init_worker,
                                  initargs=(str(images_path), train_idx, positions)) as pool:
        for count in pool.imap_unordered(_generate_chunk, tasks):
            done += count
            if done / n >= next_report:
                print(f"  {done}/{n} images ({done / n:.0%})", flush=True)
                next_report += 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N training images (smoke test / determinism check)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    root = data_dir()
    out = args.out or root / f"views_K{config.K_VIEWS}.npy"

    banner(f"E6: pre-generate K={config.K_VIEWS} views per training image")
    train_idx = load_pickle(root / "split_idx.pkl")["train"]
    positions = np.arange(len(train_idx) if args.limit is None else args.limit)
    print(f"{len(positions)} images x {config.K_VIEWS} views, {args.workers or os.cpu_count()} workers")

    start = time.perf_counter()
    with atomic_path(out) as tmp:
        generate_views(root / "eurosat" / "images.npy", train_idx, positions, tmp,
                       workers=args.workers)
    seconds = time.perf_counter() - start
    n_views = len(positions) * config.K_VIEWS
    print(f"{n_views} views in {seconds:.1f} s ({n_views / seconds:.0f} views/s)")

    write_meta(out, shape=[len(positions), config.K_VIEWS, config.IMAGE_SIZE, config.IMAGE_SIZE, 3],
               k_views=config.K_VIEWS, view_seed=config.VIEW_SEED, min_scale=config.MIN_SCALE,
               blur_sigmas=list(config.BLUR_SIGMAS), rows="positions in split_idx.pkl['train']",
               seconds=round(seconds, 1), views_per_second=round(n_views / seconds, 1),
               workers=args.workers or os.cpu_count())
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
