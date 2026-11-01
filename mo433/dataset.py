"""Step E1: download EuroSAT and freeze it into arrays, plus the fixed split lists.

Outputs, under $MO433_DATA:
  raw/eurosat/                 torchvision download (JPEGs)
  splits/eurosat-{train,val,test}.txt
                               TorchGeo split lists (Neumann et al., 2019)
  eurosat/labels.npy           (27000,) int64
  eurosat/index.json           class names and the file name of every image
  eurosat/images.npy           (27000, 64, 64, 3) uint8, written last (it is the make target)

A single array is much faster to read on Colab than 27k small files on Drive. The image order
is torchvision's: classes in alphabetical order, then file names in lexicographic order.
The split lists are checked to cover every image exactly once; step E2 turns them into indices.
"""

import argparse
import hashlib
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from mo433 import config
from mo433.io import atomic_path, banner, data_dir, save_json, write_meta


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_eurosat(raw_dir: Path):
    """Return torchvision's (path, class) samples and class names, downloading if needed."""
    from torchvision.datasets import EuroSAT
    ds = EuroSAT(root=str(raw_dir), download=True)
    return ds.samples, ds.classes


def load_images(samples):
    """Decode every JPEG into one uint8 array; also return labels and file names."""
    images = np.empty((len(samples), config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    labels = np.empty(len(samples), dtype=np.int64)
    names = []
    for i, (path, label) in enumerate(samples):
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"))
        if arr.shape != images.shape[1:]:
            raise ValueError(f"{path}: unexpected shape {arr.shape}")
        images[i], labels[i] = arr, label
        names.append(Path(path).name)
    return images, labels, names


def download_split_lists(splits_dir: Path) -> dict:
    """Download the TorchGeo split lists (if missing) and return {name: [file names]}."""
    lists = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"eurosat-{name}.txt"
        if not path.exists():
            with atomic_path(path) as tmp:
                urllib.request.urlretrieve(config.SPLIT_URL.format(name), tmp)
        lists[name] = path.read_text().split()
    return lists


def map_split_lists(filenames, lists) -> dict:
    """Map each split list to indices into ``filenames``.

    Raises ValueError unless the lists, together, cover every file exactly once.
    """
    position = {name: i for i, name in enumerate(filenames)}
    if len(position) != len(filenames):
        raise ValueError("duplicate file names in the dataset")
    listed = [n for names in lists.values() for n in names]
    duplicated = [n for n, c in Counter(listed).items() if c > 1]
    missing = sorted(set(listed) - position.keys())
    unlisted = sorted(position.keys() - set(listed))
    if duplicated or missing or unlisted:
        raise ValueError(f"split lists do not partition the dataset: {len(duplicated)} duplicated, "
                         f"{len(missing)} not in the dataset, {len(unlisted)} not in any list")
    return {split: np.array(sorted(position[n] for n in names)) for split, names in lists.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args()
    root = data_dir()
    out_dir = root / "eurosat"

    banner("E1: EuroSAT -> arrays")
    samples, classes = download_eurosat(root / "raw")
    images, labels, names = load_images(samples)
    counts = {classes[c]: int(n) for c, n in zip(*np.unique(labels, return_counts=True))}

    if len(images) != config.N_IMAGES:
        raise ValueError(f"expected {config.N_IMAGES} images, got {len(images)}")
    if counts != config.EXPECTED_CLASS_COUNTS:
        raise ValueError(f"class counts differ from the paper: {counts}")
    print(f"{len(images)} images of {images.shape[1:]} in {len(classes)} classes")
    for name, n in counts.items():
        print(f"  {name:<22} {n}")

    lists = download_split_lists(root / "splits")
    split_idx = map_split_lists(names, lists)
    print("split lists: " + ", ".join(f"{k}={len(v)}" for k, v in split_idx.items())
          + f"  -> cover all {len(names)} images exactly once")

    with atomic_path(out_dir / "labels.npy") as tmp, open(tmp, "wb") as f:
        np.save(f, labels)
    save_json({"classes": classes, "filenames": names}, out_dir / "index.json")
    with atomic_path(out_dir / "images.npy") as tmp, open(tmp, "wb") as f:
        np.save(f, images)

    write_meta(out_dir / "images.npy",
               n_images=len(images), shape=list(images.shape), class_counts=counts,
               sha256={"images.npy": sha256_file(out_dir / "images.npy"),
                       "labels.npy": sha256_file(out_dir / "labels.npy"),
                       **{f"eurosat-{k}.txt": sha256_file(root / "splits" / f"eurosat-{k}.txt")
                          for k in lists}},
               split_list_sizes={k: len(v) for k, v in lists.items()},
               split_source=config.SPLIT_URL)
    print(f"✓ written to {out_dir}")


if __name__ == "__main__":
    main()
