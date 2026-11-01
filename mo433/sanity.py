"""Sanity and collapse checks (steps E9-E10), written to $MO433_DATA/sanity/<encoder>.json.

Encoders available here:
  pixels       raw pixels, flattened (12,288-d): a reference point that needs no network.
               Also runs the k-NN with shuffled training labels, which must land near chance.

The teacher and its negative control are added in step E10, and the students later.
"""

import argparse

import numpy as np

from mo433 import config
from mo433.evaluate import fit_preprocessor, knn_accuracy, sanity_diagnostics
from mo433.io import banner, data_dir, load_pickle, save_json, write_meta

ENCODERS = ("pixels",)


def load_split_data(root):
    images = np.load(root / "eurosat" / "images.npy", mmap_mode="r")
    labels = np.load(root / "eurosat" / "labels.npy")
    split = load_pickle(root / "split_idx.pkl")
    return images, labels, split


def pixel_features(images, idx) -> np.ndarray:
    return np.asarray(images[idx], dtype=np.float32).reshape(len(idx), -1) / 255.0


def shuffled_label_knn(train_feats, y_train, test_feats, y_test, seed=0) -> float:
    """k-NN accuracy after shuffling the training labels: must be near chance (10%)."""
    pre = fit_preprocessor(train_feats)
    y_shuffled = np.random.default_rng(seed).permutation(y_train)
    return knn_accuracy(pre.transform(train_feats), y_shuffled, pre.transform(test_feats),
                        y_test)["accuracy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("encoder", choices=ENCODERS)
    args = parser.parse_args()
    root = data_dir()
    out = root / "sanity" / f"{args.encoder}.json"

    banner(f"Sanity check: {args.encoder}")
    images, labels, split = load_split_data(root)
    y_train, y_test = labels[split["train"]], labels[split["test"]]
    subsets = load_pickle(root / "label_subsets.pkl")

    train_feats = pixel_features(images, split["train"])
    test_feats = pixel_features(images, split["test"])
    result = sanity_diagnostics(train_feats, y_train, test_feats, y_test, subsets)
    result["knn_100_shuffled_labels"] = shuffled_label_knn(train_feats, y_train, test_feats, y_test)
    result["chance"] = 1 / config.NUM_CLASSES

    save_json(result, out)
    write_meta(out, encoder=args.encoder)
    for k, v in result.items():
        print(f"  {k:<30} {v}")
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
