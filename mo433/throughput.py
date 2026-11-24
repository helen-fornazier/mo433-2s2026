"""Step E14: throughput of every training stage on the fixed GPU (spec 3.9; report Sec. 5.3).

Writes $MO433_DATA/throughput.json. For each candidate batch size, in images per second:

  loader_on_the_fly  DataLoader applying lightly's VICRegTransform (two views per image) to the
                     training images, with one worker per CPU core, over one full epoch: the
                     option the fixed views replace;
  loader_views       gathering a batch from the pre-generated views held on the GPU (ViewBank);
  step_B_c<c>        a full training step of row B (gather + forward + backward + AdamW), with
                     the largest head of the tuning grid;
  step_C_c<c>        idem for row C, with the cached teacher targets;
  teacher_forward    the frozen teacher at 224 px in fp16 (as in step E13, without disk reading).

From the step times it tabulates the GPU-hours of the whole study (tuning + the 20 main runs,
the 50% runs at half the steps) for a few epoch budgets, to fix epochs and batch size.
"""

import argparse
import os
import time

import numpy as np
import torch

from mo433 import config
from mo433.io import banner, data_dir, host_label, load_pickle, save_json, write_meta
from mo433.teacher import load_teacher
from mo433.teacher_cache import COLAB_STAGE_DIR, stage
from mo433.training import ViewBank, create_row, epoch_batches, steps_per_epoch, train_step


def timed(fn, warmup: int, iters: int, device) -> float:
    """Seconds per call of ``fn``, after ``warmup`` calls, synchronizing the GPU."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


class OnTheFlyDataset(torch.utils.data.Dataset):
    def __init__(self, images):
        from mo433.transforms import VIEW_KWARGS
        from lightly.transforms import VICRegTransform
        self.images, self.transform = images, VICRegTransform(**VIEW_KWARGS)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        from PIL import Image
        return self.transform(Image.fromarray(self.images[i]))


def loader_on_the_fly(images, batch_size: int) -> float:
    """Images per second over one full epoch. The workers are started by a first, untimed
    epoch; timing a few batches instead would count prefetched batches as free."""
    loader = torch.utils.data.DataLoader(OnTheFlyDataset(images), batch_size=batch_size,
                                         shuffle=True, num_workers=os.cpu_count(),
                                         drop_last=True, persistent_workers=True)
    for _ in loader:
        pass
    start = time.perf_counter()
    n = sum(len(views[0]) for views in loader)
    return n / (time.perf_counter() - start)


def batch_stream(bank, n_views: int, batch_size: int, seed: int = 0):
    """Endless (idx, k) batches over all training positions, as in training."""
    g = torch.Generator().manual_seed(seed)
    while True:
        yield from epoch_batches(np.arange(bank.n_images), bank.k_views, batch_size, n_views, g)


def measure_step(bank, row: str, c: int, batch_size: int, warmup: int, iters: int) -> float:
    model = create_row(row, c, config.THROUGHPUT_HIDDEN, seed=0).to(bank.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                  weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(bank.device.type, enabled=bank.device.type == "cuda")
    stream = batch_stream(bank, model.n_views, batch_size)

    def step():
        train_step(model, optimizer, scaler, bank, *next(stream))
    return batch_size / timed(step, warmup, iters, bank.device)


def budget(results: dict, batch_size: int) -> dict:
    """GPU-hours of tuning + main runs for each epoch option, from the measured step rates."""
    n = config.N_TRAIN
    epoch_s = {(row, c): steps_per_epoch(n, batch_size) * batch_size
               / results[f"step_{row}_c{c}"] for row in "BC" for c in config.WIDTHS}
    tuning_s = sum(len(config.TUNING_LRS) * len(config.TUNING_HIDDEN) * config.TUNING_EPOCHS
                   * epoch_s[(row, config.TUNING_WIDTH)] for row in "BC")
    out = {"epoch_seconds": {f"{row}_c{c}": round(s, 1) for (row, c), s in epoch_s.items()},
           "tuning_gpu_hours": round(tuning_s / 3600, 2), "by_epochs": {}}
    for epochs in config.THROUGHPUT_EPOCH_OPTIONS:
        # main runs: every seed at 100% and at 50% of the images (half the steps)
        main_s = sum(len(config.SEEDS[c]) * (1 + 0.5) * epochs * epoch_s[(row, c)]
                     for row in "BC" for c in config.WIDTHS)
        out["by_epochs"][epochs] = {"main_gpu_hours": round(main_s / 3600, 2),
                                    "total_gpu_hours": round((main_s + tuning_s) / 3600, 2),
                                    "longest_run_minutes": round(
                                        epochs * max(epoch_s.values()) / 60, 1)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true",
                        help="tiny run on 1,024 images, few iterations, one batch size (smoke test)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = data_dir()
    out = args.out or root / "throughput.json"
    device = torch.device(args.device)
    banner(f"E14: throughput ({args.device})")
    if device.type == "cpu" and not args.quick:
        raise SystemExit("no GPU: the measurement runs on Colab (or pass --quick for a smoke test)")

    warmup, iters = (1, 2) if args.quick else (config.THROUGHPUT_WARMUP, config.THROUGHPUT_ITERS)
    batches = (64,) if args.quick else config.THROUGHPUT_BATCHES
    limit = 1024 if args.quick else None

    stage_dir = COLAB_STAGE_DIR if host_label() == "colab" else None
    views = np.load(stage(root / f"views_K{config.K_VIEWS}.npy", stage_dir), mmap_mode="r")
    targets = np.load(root / f"teacher_feats_K{config.K_VIEWS}.npy", mmap_mode="r")
    images = np.load(root / "eurosat" / "images.npy")[load_pickle(root / "split_idx.pkl")["train"]]
    start = time.perf_counter()
    bank = ViewBank(views[:limit], targets[:limit], device)
    print(f"  views and targets on {device} in {time.perf_counter() - start:.0f} s")
    teacher = load_teacher(device)

    results = {}
    for bs in batches:
        r = {"loader_on_the_fly": loader_on_the_fly(images[:limit], bs)}
        stream = batch_stream(bank, 1, bs)
        r["loader_views"] = bs / timed(lambda: bank.images(*next(stream)).sum(), warmup,
                                       iters, device)
        for row in "BC":
            for c in config.WIDTHS:
                r[f"step_{row}_c{c}"] = measure_step(bank, row, c, bs, warmup, iters)
        r["teacher_forward"] = bs / timed(lambda: teacher(bank.images(*next(stream))),
                                          warmup, iters, device)
        r = {k: round(v, 1) for k, v in r.items()}
        r["budget"] = budget(r, bs)
        results[bs] = r
        print(f"batch {bs}:")
        for k, v in r.items():
            print(f"  {k:<20} {v}")

    save_json(results, out)
    write_meta(out, device=str(device), cpu_workers=os.cpu_count(), quick=args.quick,
               warmup=warmup, iters=iters, hidden=config.THROUGHPUT_HIDDEN)
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
