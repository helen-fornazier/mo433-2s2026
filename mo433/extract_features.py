"""Extract and cache the features of one encoder (spec 3.6), with the single ``extract`` function.

Writes $MO433_DATA/feats/<encoder>_train.npy and <encoder>_test.npy, (N, d) float32, in the
order of split_idx.pkl["train"] / ["test"]. Images go through the deterministic evaluation
path only (each encoder module does its own normalization / resizing).

Encoders available so far:
  D          the frozen DINO ViT-S/16 teacher (row D)
  D-random   the same ViT with random weights (seed 0): negative control for the sanity check,
             not one of the compared encoders
Rows A, B and C are added in later steps.
"""

import argparse
import time

import numpy as np
import torch

from mo433.features import extract
from mo433.io import atomic_path, banner, data_dir, load_pickle, write_meta
from mo433.teacher import load_teacher, weights_sha256

ENCODERS = ("D", "D-random")


def build_encoder(name: str, device="cpu"):
    if name == "D":
        return load_teacher(device)
    if name == "D-random":
        torch.manual_seed(0)
        return load_teacher(device, pretrained=False)
    raise ValueError(f"unknown encoder {name!r}")


def feature_paths(root, name):
    return {split: root / "feats" / f"{name}_{split}.npy" for split in ("train", "test")}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoder", choices=ENCODERS)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    root = data_dir()

    banner(f"Extract features: {args.encoder} ({args.device})")
    images = np.load(root / "eurosat" / "images.npy", mmap_mode="r")
    split = load_pickle(root / "split_idx.pkl")
    model = build_encoder(args.encoder, args.device)

    # train first, test last: the test file is the make target
    for name in ("train", "test"):
        out = feature_paths(root, args.encoder)[name]
        start = time.perf_counter()
        feats = extract(model, images[split[name]], batch_size=args.batch_size, device=args.device)
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
