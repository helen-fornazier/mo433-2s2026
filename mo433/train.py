"""Step E15: train one student of row B or C (spec 3.4, 3.5, 3.9; report Secs. 3.5, 3.6, 5.2).

    python -m mo433.train <name> [--lr LR] [--hidden H] [--epochs N] [--tag TAG]

The run name says what is trained:
  <row>-c<c>-d<percent>-s<seed>                 a main run: row B (VICReg) or C (distillation),
                                                student width c, percentage of the unlabeled
                                                images (unlab_idx_<percent>.pkl), seed; lr and h
                                                are the row's tuned values (config.SELECTED)
  <row>-c<c>-d<percent>-s<seed>-tune-lr<lr>-h<h>  a tuning trial (step E17): that lr and h, for
                                                TUNING_EPOCHS epochs
--lr, --hidden and --epochs override these; ``--tag`` appends -<tag> to the name (smoke tests).

If the loss stops being finite, a main run aborts and writes nothing. A tuning trial instead
stops there and is written as diverged (``diverged_at_epoch`` in its log and metadata): a learning
rate that diverges is a result of the tuning, and the trial must exist for the tuning to go on.

Both rows use the same budget: the same epochs, batch size (last incomplete batch dropped, so
the same number of steps), AdamW, weight decay, and a linear warm-up over WARMUP_FRACTION of the
steps followed by a cosine decay to zero, all per step. The seed fixes the student's
initialization (identical to row A's encoder with the same seed) and the view sampler.
cuDNN kernels are not bitwise deterministic, so two runs with the same seed can differ slightly.

Writes, only when training ends (the checkpoint of the last epoch, report Sec. 3.5):
  runs/<name>.pth        the student's state_dict (the head / projector is discarded)
  runs/<name>_log.json   loss, learning rate, seconds and skipped steps of each epoch
and the usual .json metadata with the GPU, the wall time and the GPU-hours.
"""

import argparse
import math
import re
import time
from pathlib import Path

import numpy as np
import torch

from mo433 import config
from mo433.io import atomic_path, banner, data_dir, load_pickle, save_json, write_meta
from mo433.teacher import weights_sha256
from mo433.training import ViewBank, create_row, epoch_batches, steps_per_epoch, train_step

RUN = re.compile(r"([BC])-c(\d+)-d(\d+)-s(\d+)(?:-tune-lr([0-9.e+-]+)-h(\d+))?")


def parse_run(run: str) -> dict:
    """Row, width, data percentage, seed, and the lr / h / epochs the name implies."""
    m = RUN.fullmatch(run)
    if not m:
        raise ValueError(f"run name must look like B-c16-d100-s0[-tune-lr1e-3-h2048], got {run!r}")
    out = {"row": m[1], "c": int(m[2]), "percent": int(m[3]), "seed": int(m[4])}
    if m[5]:
        out.update(lr=float(m[5]), hidden=int(m[6]), epochs=config.TUNING_EPOCHS, tuning=True)
    else:
        selected = config.SELECTED[out["row"]]
        lr, hidden = selected if selected else (None, None)
        out.update(lr=lr, hidden=hidden, epochs=config.EPOCHS, tuning=False)
    return out


def main_runs(row: str, c: int | None = None) -> list:
    """The main runs of a row (all widths, or width ``c``): every seed at every unlabeled
    percentage, each block (row x width) with the 50% runs first (spec 5: reduced data first)."""
    widths = config.WIDTHS if c is None else (c,)
    return [f"{row}-c{w}-d{p}-s{s}" for w in widths for p in sorted(config.UNLAB_FRACTIONS)
            for s in config.SEEDS[w]]


def steps_taken(optimizer) -> int:
    """Optimizer steps actually applied. The gradient scaler skips the step of a batch whose
    gradients overflowed in fp16; AdamW counts only the steps it takes (a CPU tensor: no sync)."""
    state = optimizer.state.get(optimizer.param_groups[0]["params"][0], {})
    return int(state["step"]) if "step" in state else 0


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up from 1/warmup_steps to 1, then cosine decay to 0 at the last step."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train(row: str, c: int, seed: int, positions, bank: ViewBank, *, lr: float, hidden: int,
          epochs: int, batch_size: int = config.BATCH_SIZE, stop_if_diverged: bool = False,
          log=print):
    """Train one student; returns (student, per-epoch log). A non-finite epoch loss raises, or,
    with ``stop_if_diverged``, ends the training with ``diverged_at_epoch`` in the log."""
    model = create_row(row, c, hidden, seed).to(bank.device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
    n_steps = steps_per_epoch(len(positions), batch_size)
    total = epochs * n_steps
    warmup = max(1, round(config.WARMUP_FRACTION * total))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda s: lr_factor(s, total, warmup))
    scaler = torch.amp.GradScaler(bank.device.type, enabled=bank.device.type == "cuda")
    generator = torch.Generator().manual_seed(seed)

    history, diverged_at, skipped = [], None, 0
    for epoch in range(epochs):
        start, loss_sum = time.perf_counter(), torch.zeros((), device=bank.device)
        epoch_lr = scheduler.get_last_lr()[0]
        for idx, k in epoch_batches(positions, bank.k_views, batch_size, model.n_views,
                                    generator):
            loss_sum += train_step(model, optimizer, scaler, bank, idx, k)
            scheduler.step()
        loss = (loss_sum / n_steps).item()
        if not math.isfinite(loss):
            if not stop_if_diverged:
                raise RuntimeError(f"loss is {loss} at epoch {epoch + 1}; nothing written")
            diverged_at = epoch + 1
            log(f"  epoch {epoch + 1:>3}/{epochs}  loss {loss}: diverged, training stopped")
            break
        epoch_skipped = (epoch + 1) * n_steps - steps_taken(optimizer) - skipped
        skipped += epoch_skipped
        history.append({"epoch": epoch + 1, "loss": round(loss, 5), "lr_start": epoch_lr,
                        "seconds": round(time.perf_counter() - start, 2),
                        "skipped_steps": epoch_skipped})
        log(f"  epoch {epoch + 1:>3}/{epochs}  loss {loss:.4f}  lr {epoch_lr:.2e}  "
            f"{history[-1]['seconds']:.1f} s" + (f"  ({epoch_skipped} steps skipped)"
                                                 if epoch_skipped else ""))
    out = {"steps_per_epoch": n_steps, "total_steps": total, "warmup_steps": warmup,
           "skipped_steps": skipped, "epochs": history}
    if diverged_at is not None:
        out["diverged_at_epoch"] = diverged_at
    return model.student, out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="e.g. B-c16-d100-s0 or B-c16-d100-s0-tune-lr1e-3-h2048")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden", type=int, default=None, help="head / projector width h")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tag", default=None, help="suffix of the run name (tuning, checks)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N training images (smoke test), written to --out-dir")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run = parse_run(args.run)
    for key in ("lr", "hidden", "epochs"):
        if getattr(args, key) is not None:
            run[key] = getattr(args, key)
    if run["lr"] is None:
        raise SystemExit(f"no lr / h for row {run['row']}: run the tuning (E17) or pass --lr --hidden")
    name = args.run + (f"-{args.tag}" if args.tag else "")
    root = data_dir()
    out_dir = (args.out_dir or root) / "runs"

    banner(f"E15: train {name} (lr {run['lr']}, h {run['hidden']}, {run['epochs']} epochs, {args.device})")
    if args.device == "cpu" and args.limit is None:
        raise SystemExit("no GPU: trainings run on Colab (use --limit for a smoke test)")
    positions = load_pickle(root / f"unlab_idx_{run['percent']}.pkl")
    views = np.load(root / f"views_K{config.K_VIEWS}.npy", mmap_mode="r")
    targets = (np.load(root / f"teacher_feats_K{config.K_VIEWS}.npy", mmap_mode="r")
               if run["row"] == "C" else None)
    if args.limit is not None:
        positions = positions[positions < args.limit]
        views = views[:args.limit]
        targets = None if targets is None else targets[:args.limit]
    bank = ViewBank(views, targets, args.device)
    gpu = torch.cuda.get_device_name() if bank.device.type == "cuda" else None
    if gpu is not None and "L4" not in gpu:
        print(f"WARNING: running on {gpu}, not the L4 fixed for all timed runs (report Sec. 5.3)")

    start = time.perf_counter()
    student, history = train(run["row"], run["c"], run["seed"], positions, bank, lr=run["lr"],
                             hidden=run["hidden"], epochs=run["epochs"],
                             batch_size=min(config.BATCH_SIZE, len(positions)),
                             stop_if_diverged=run["tuning"])
    seconds = time.perf_counter() - start

    weights = out_dir / f"{name}.pth"
    info = dict(run, name=name, n_images=len(positions),
                batch_size=min(config.BATCH_SIZE, len(positions)), limit=args.limit,
                weight_decay=config.WEIGHT_DECAY, warmup_fraction=config.WARMUP_FRACTION,
                device=args.device, gpu=gpu, seconds=round(seconds, 1),
                gpu_hours=round(seconds / 3600, 4) if gpu else None)
    if "diverged_at_epoch" in history:
        info["diverged_at_epoch"] = history["diverged_at_epoch"]
    save_json(dict(info, **history), out_dir / f"{name}_log.json")
    with atomic_path(weights) as tmp:
        torch.save({k: v.cpu() for k, v in student.state_dict().items()}, tmp)
    final_loss = history["epochs"][-1]["loss"] if history["epochs"] else None
    write_meta(weights, **info, final_loss=final_loss, skipped_steps=history["skipped_steps"],
               weights_sha256=weights_sha256(student))
    if "diverged_at_epoch" in history:
        print(f"✗ {name}: diverged at epoch {history['diverged_at_epoch']}, recorded as such in "
              f"{weights} and its log")
        return
    print(f"✓ {name}: {seconds / 60:.1f} min, final loss {final_loss:.4f}, written to {weights}")


if __name__ == "__main__":
    main()
