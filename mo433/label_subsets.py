"""Step E3: labeled subsets for the probes, saved as $MO433_DATA/label_subsets.pkl.

{(fraction, draw): indices} for fractions 1%, 10%, 50%, 100% and draws 0, 1, 2, stratified by
class. The indices are positions within the training split (split_idx.pkl["train"]), i.e.
into y_train = labels[split["train"]]. The subsets are generated once, here, and every encoder
is probed on exactly these subsets (spec 3.1). ``make_label_subsets`` is the code of the
specification, unchanged apart from being wrapped in a function.
"""

import argparse

import numpy as np
from sklearn.model_selection import train_test_split

from mo433 import config
from mo433.io import banner, data_dir, load_pickle, save_pickle, write_meta


def make_label_subsets(y_train, fractions=config.LABEL_FRACTIONS, draws=config.LABEL_DRAWS) -> dict:
    subsets = {}
    for frac in fractions:
        for draw in draws:  # 3 independent labeled subsets
            if frac == 1.0:
                subsets[(frac, draw)] = np.arange(len(y_train))
            else:
                idx, _ = train_test_split(
                    np.arange(len(y_train)), train_size=frac,
                    stratify=y_train, random_state=draw)
                subsets[(frac, draw)] = idx
    return subsets


def knn_k(y_subset) -> int:
    """k of the k-NN probe: min(20, number of examples of the smallest class in the subset)."""
    return int(min(config.KNN_MAX_K, np.bincount(y_subset).min()))


def summarize(subsets, y_train) -> dict:
    """Per subset: size, smallest/largest class count, k of the k-NN probe."""
    out = {}
    for (frac, draw), idx in subsets.items():
        counts = np.bincount(y_train[idx], minlength=config.NUM_CLASSES)
        out[f"{frac:.2f}/draw{draw}"] = {"size": len(idx), "smallest_class": int(counts.min()),
                                         "largest_class": int(counts.max()), "knn_k": knn_k(y_train[idx])}
    return out


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    root = data_dir()
    out = root / "label_subsets.pkl"

    banner("E3: labeled subsets")
    y_train = np.load(root / "eurosat" / "labels.npy")[load_pickle(root / "split_idx.pkl")["train"]]
    subsets = make_label_subsets(y_train)
    summary = summarize(subsets, y_train)
    save_pickle(subsets, out)
    write_meta(out, fractions=list(config.LABEL_FRACTIONS), draws=list(config.LABEL_DRAWS),
               indices_into="split_idx.pkl['train']", summary=summary)

    print(f"{'subset':<12} {'size':>6} {'smallest':>9} {'largest':>8} {'k':>3}")
    for name, s in summary.items():
        print(f"{name:<12} {s['size']:>6} {s['smallest_class']:>9} {s['largest_class']:>8} {s['knn_k']:>3}")
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
