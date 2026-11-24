"""Step E1b: descriptive statistics of EuroSAT, written to $MO433_DATA/eurosat/stats.json.

The numbers that go into the report (class counts per split, per-channel statistics, duplicate
and degenerate images) are computed here, so that they do not depend on anyone running the
exploration notebook (notebooks/explore_eurosat.ipynb), which imports these functions for its
figures.

Splits here are the raw TorchGeo lists (train / val / test); our training split is
train + val (report Sec. 5.1).
"""

import argparse
import hashlib
from collections import defaultdict

import numpy as np

from mo433 import config
from mo433.dataset import load_eurosat
from mo433.io import banner, data_dir, save_json, write_meta

# An image is "constant" if its pixel standard deviation (uint8 scale) is below this value,
# and "near black" / "near white" if its mean is below / above these values.
CONSTANT_STD = 1.0
DARK_MEAN, BRIGHT_MEAN = 5.0, 250.0


def split_groups(lists) -> dict:
    """Index sets used in the report: the three TorchGeo lists and our train/test split."""
    return {"all": None, "torchgeo_train": lists["train"], "torchgeo_val": lists["val"],
            "torchgeo_test": lists["test"],
            "train (train+val)": np.sort(np.concatenate([lists["train"], lists["val"]])),
            "test": lists["test"]}


def class_counts(labels, classes, groups) -> dict:
    """{group: {class: count}} for every group of indices (None = whole dataset)."""
    out = {}
    for name, idx in groups.items():
        y = labels if idx is None else labels[idx]
        out[name] = {c: int((y == k).sum()) for k, c in enumerate(classes)}
    return out


def max_proportion_gap(counts, reference="all") -> dict:
    """Largest absolute difference (percentage points) between a group's class proportions
    and the reference group's; 0 means perfectly stratified."""
    ref = np.array(list(counts[reference].values()), dtype=float)
    ref /= ref.sum()
    gaps = {}
    for name, c in counts.items():
        p = np.array(list(c.values()), dtype=float)
        gaps[name] = round(float(np.abs(p / p.sum() - ref).max() * 100), 3)
    return gaps


def channel_stats(images) -> dict:
    """Per-channel mean and standard deviation in [0, 1], as used by Normalize()."""
    x = images.reshape(-1, 3).astype(np.float64) / 255.0
    return {"mean": x.mean(0).round(4).tolist(), "std": x.std(0).round(4).tolist()}


def per_class_channel_mean(images, labels, classes) -> dict:
    return {c: (images[labels == k].reshape(-1, 3).mean(0) / 255.0).round(4).tolist()
            for k, c in enumerate(classes)}


def exact_duplicates(images) -> list:
    """Groups (lists of indices) of images whose pixels are byte-for-byte identical."""
    by_hash = defaultdict(list)
    for i, img in enumerate(images):
        by_hash[hashlib.sha1(img.tobytes()).hexdigest()].append(i)
    return [g for g in by_hash.values() if len(g) > 1]


def cross_split_duplicates(groups_of_dups, train_idx, test_idx) -> list:
    """Duplicate groups that have members on both sides of the train/test split."""
    train, test = set(train_idx.tolist()), set(test_idx.tolist())
    return [g for g in groups_of_dups if train & set(g) and test & set(g)]


def degenerate_images(images) -> dict:
    flat = images.reshape(len(images), -1).astype(np.float32)
    std, mean = flat.std(1), flat.mean(1)
    return {"constant": np.flatnonzero(std < CONSTANT_STD).tolist(),
            "near_black": np.flatnonzero(mean < DARK_MEAN).tolist(),
            "near_white": np.flatnonzero(mean > BRIGHT_MEAN).tolist()}


def compute_stats(data) -> dict:
    images, labels, classes, lists = data["images"], data["labels"], data["classes"], data["lists"]
    groups = split_groups(lists)
    counts = class_counts(labels, classes, groups)
    dups = exact_duplicates(images)
    train_idx = groups["train (train+val)"]
    return {
        "class_counts": counts,
        "max_proportion_gap_pp": max_proportion_gap(counts),
        "channel_stats": channel_stats(images),
        "channel_stats_train": channel_stats(images[train_idx]),
        "imagenet": {"mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD)},
        "per_class_channel_mean": per_class_channel_mean(images, labels, classes),
        "exact_duplicate_groups": dups,
        "cross_split_duplicate_groups": cross_split_duplicates(dups, train_idx, groups["test"]),
        "degenerate": degenerate_images(images),
        "thresholds": {"constant_std": CONSTANT_STD, "dark_mean": DARK_MEAN,
                       "bright_mean": BRIGHT_MEAN},
    }


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = data_dir() / "eurosat" / "stats.json"

    banner("E1b: EuroSAT statistics")
    stats = compute_stats(load_eurosat())
    save_json(stats, out)
    write_meta(out)

    gaps = stats["max_proportion_gap_pp"]
    for name in ("train (train+val)", "test"):
        print(f"{name:<18} max class-proportion gap vs. full dataset: {gaps[name]:.2f} pp")
    cs, inet = stats["channel_stats"], stats["imagenet"]
    print(f"channel mean  EuroSAT {cs['mean']}  ImageNet {inet['mean']}")
    print(f"channel std   EuroSAT {cs['std']}  ImageNet {inet['std']}")
    print(f"exact duplicate groups: {len(stats['exact_duplicate_groups'])} "
          f"(across train/test: {len(stats['cross_split_duplicate_groups'])})")
    print("degenerate images: " + ", ".join(f"{k}={len(v)}" for k, v in stats["degenerate"].items()))
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
