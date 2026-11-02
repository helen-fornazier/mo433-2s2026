"""Extract and cache the features of one encoder (spec 3.6), with the single ``extract`` function.

Writes $MO433_DATA/feats/<encoder>_train.npy and <encoder>_test.npy, (N, d) float32, in the
order of split_idx.pkl["train"] / ["test"]. Images go through the deterministic evaluation
path only (each encoder module does its own normalization / resizing).

Encoders available so far:
  D            the frozen DINO ViT-S/16 teacher (row D)
  D-random     the same ViT with random weights (seed 0): negative control for the sanity
               check, not one of the compared encoders
  A-c<c>-s<s>  row A: the untrained student of width c with the initialization of seed s,
               BatchNorm in eval mode with its default running statistics (spec 2.1). Its
               weights are also saved to runs/A-c<c>-s<s>.pth (for the latency measurement).
  <B|C>-c<c>-d<p>-s<s>[-<tag>]
               a trained student of row B or C, loaded from runs/<name>.pth (step E15).
"""

import argparse
import re
import time

from pathlib import Path

import numpy as np
import torch

from mo433.features import extract
from mo433.io import atomic_path, banner, data_dir, load_pickle, write_meta
from mo433.student import create_student
from mo433.teacher import load_teacher, weights_sha256

ROW_A = re.compile(r"A-c(\d+)-s(\d+)")
TRAINED = re.compile(r"[BC]-c(\d+)-d\d+-s(\d+)(?:-.+)?")


def build_encoder(name: str, device="cpu", root=None):
    if name == "D":
        return load_teacher(device)
    if name == "D-random":
        torch.manual_seed(0)
        return load_teacher(device, pretrained=False)
    if m := ROW_A.fullmatch(name):
        return create_student(int(m[1]), seed=int(m[2])).to(device)
    if m := TRAINED.fullmatch(name):
        student = create_student(int(m[1]), seed=int(m[2]))
        student.load_state_dict(torch.load((root or data_dir()) / "runs" / f"{name}.pth",
                                           map_location="cpu"))
        return student.to(device)
    raise ValueError(f"unknown encoder {name!r}")


def feature_paths(root, name):
    return {split: root / "feats" / f"{name}_{split}.npy" for split in ("train", "test")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoder", help="D, D-random, A-c<c>-s<s> or a trained run")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N images of each split, written to --out-dir "
                             "(to check that a code change leaves the features unchanged)")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = data_dir()

    banner(f"Extract features: {args.encoder} ({args.device})")
    images = np.load(root / "eurosat" / "images.npy", mmap_mode="r")
    split = load_pickle(root / "split_idx.pkl")
    model = build_encoder(args.encoder, args.device, root)
    if args.limit is None and ROW_A.fullmatch(args.encoder):
        weights = root / "runs" / f"{args.encoder}.pth"
        with atomic_path(weights) as tmp:
            torch.save(model.state_dict(), tmp)
        write_meta(weights, encoder=args.encoder, weights_sha256=weights_sha256(model))

    # train first, test last: the test file is the make target
    for name in ("train", "test"):
        out = feature_paths(args.out_dir or root, args.encoder)[name]
        idx = split[name] if args.limit is None else split[name][:args.limit]
        start = time.perf_counter()
        feats = extract(model, images[idx], batch_size=args.batch_size, device=args.device)
        seconds = time.perf_counter() - start
        with atomic_path(out) as tmp, open(tmp, "wb") as f:
            np.save(f, feats)
        write_meta(out, encoder=args.encoder, split=name, shape=list(feats.shape),
                   device=args.device, batch_size=args.batch_size, seconds=round(seconds, 1),
                   images_per_second=round(len(feats) / seconds, 1),
                   weights_sha256=weights_sha256(model))
        print(f"  {name:<5} {feats.shape}  {seconds:.0f} s ({len(feats) / seconds:.0f} img/s)")
    print(f"✓ written to {root / 'feats'}")


if __name__ == "__main__":
    main()
