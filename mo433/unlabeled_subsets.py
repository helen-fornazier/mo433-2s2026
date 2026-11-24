"""Step E4: unlabeled training subsets for rows B and C, saved as
$MO433_DATA/unlab_idx_100.pkl and unlab_idx_50.pkl.

Indices are positions within the training split (split_idx.pkl["train"]), which is also the
order of the pre-generated views (step E6). 100% is the whole training split; 50% is the
ablation level (spec 5), a uniform draw without replacement with seed 0 (report Sec. 6.5).
The draw never sees the labels: ``draw_unlabeled_subset`` only receives the number of images.
The class distribution of the 50% subset is computed afterwards, for information only.
"""

import argparse

import numpy as np

from mo433 import config
from mo433.io import banner, data_dir, load_pickle, save_pickle, write_meta


def draw_unlabeled_subset(n: int, percent: int, seed: int = config.UNLAB_SEED) -> np.ndarray:
    """Sorted indices of ``percent``% of ``n`` images, drawn uniformly without replacement."""
    if percent == 100:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=n * percent // 100, replace=False)).astype(np.int64)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    root = data_dir()
    train_idx = load_pickle(root / "split_idx.pkl")["train"]

    banner("E4: unlabeled subsets")
    subsets = {p: draw_unlabeled_subset(len(train_idx), p) for p in config.UNLAB_FRACTIONS}
    if not np.isin(subsets[50], subsets[100]).all():
        raise ValueError("the 50% subset is not contained in the 100% subset")

    # For information only: class distribution of each subset (labels read after the draw).
    y_train = np.load(root / "eurosat" / "labels.npy")[train_idx]
    for p in sorted(subsets, reverse=True):     # 100 first, 50 last (the make target)
        idx = subsets[p]
        out = root / f"unlab_idx_{p}.pkl"
        save_pickle(idx, out)
        counts = np.bincount(y_train[idx], minlength=config.NUM_CLASSES)
        write_meta(out, percent=p, size=len(idx), seed=config.UNLAB_SEED if p < 100 else None,
                   indices_into="split_idx.pkl['train']",
                   class_proportions_pct=(counts / len(idx) * 100).round(2).tolist())
        print(f"{p:>3}%: {len(idx)} images  -> {out.name}")

    full = np.bincount(y_train, minlength=config.NUM_CLASSES) / len(y_train)
    half = np.bincount(y_train[subsets[50]], minlength=config.NUM_CLASSES) / len(subsets[50])
    print(f"50% subset: max class-proportion gap vs. training split "
          f"{np.abs(half - full).max() * 100:.2f} pp (informative; drawn without labels)")
    print("✓ done")


if __name__ == "__main__":
    main()
