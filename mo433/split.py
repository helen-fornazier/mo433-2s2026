"""Step E2: the fixed train/test split, saved as $MO433_DATA/split_idx.pkl.

{"train": indices, "test": indices} into eurosat/images.npy (sorted int64 arrays).
train = TorchGeo train + val lists, test = TorchGeo test list (Neumann et al., 2019;
DECISOES 6). No randomness: the lists fix the split. The test indices are never used for
training, distillation, probe fitting or hyperparameter selection.
"""

import argparse
import hashlib
import pickle

import numpy as np

from mo433 import config
from mo433.dataset import load_eurosat
from mo433.io import banner, data_dir, save_pickle, write_meta


def make_split(lists) -> dict:
    """Merge the TorchGeo lists into our split (see config.SPLIT_LISTS)."""
    split = {name: np.sort(np.concatenate([lists[part] for part in parts])).astype(np.int64)
             for name, parts in config.SPLIT_LISTS.items()}
    if np.intersect1d(split["train"], split["test"]).size:
        raise ValueError("train and test overlap")
    return split


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    out = data_dir() / "split_idx.pkl"

    banner("E2: train/test split")
    data = load_eurosat()
    split = make_split(data["lists"])
    n_total = len(data["labels"])
    if (len(split["train"]), len(split["test"])) != (config.N_TRAIN, config.N_TEST):
        raise ValueError(f"unexpected sizes: {len(split['train'])} / {len(split['test'])}")
    if len(split["train"]) + len(split["test"]) != n_total:
        raise ValueError("the split does not cover the dataset")

    counts = {name: {c: int((data["labels"][idx] == k).sum()) for k, c in enumerate(data["classes"])}
              for name, idx in split.items()}
    save_pickle(split, out)
    write_meta(out, sizes={k: len(v) for k, v in split.items()}, class_counts=counts,
               sha256=hashlib.sha256(pickle.dumps(split)).hexdigest(),
               source="TorchGeo lists: train+val -> train, test -> test")

    for name, idx in split.items():
        c = counts[name]
        print(f"{name:<5} {len(idx):>6}  smallest {min(c, key=c.get)} ({min(c.values())}), "
              f"largest {max(c, key=c.get)} ({max(c.values())})")
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
