"""Figure: a few random EuroSAT images per class (report appendix, fig:eurosat-samples).

One column per class, ``--per-class`` rows, images shown at native resolution without
smoothing. Drawn from the whole dataset with a fixed seed, so the figure is reproducible.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mo433.dataset import load_eurosat
from mo433.io import REPO_DIR, atomic_path, banner, write_meta


def two_line_name(name: str) -> str:
    """'HerbaceousVegetation' -> 'Herbaceous\\nVegetation' (CamelCase split for column titles)."""
    return "\n".join(re.findall(r"[A-Z][a-z]*", name)) if len(name) > 8 else name


def plot_sample_grid(images, labels, classes, per_class=4, seed=0):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(per_class, len(classes),
                             figsize=(len(classes) * 0.8, per_class * 0.8 + 0.45))
    for k, name in enumerate(classes):
        picks = rng.choice(np.flatnonzero(labels == k), per_class, replace=False)
        for row, i in enumerate(picks):
            ax = axes[row, k]
            ax.imshow(images[i], interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0, k].set_title(two_line_name(name), fontsize=7, va="bottom")
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.85, wspace=0.05, hspace=0.05)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=REPO_DIR.parent / "relatorio" / "figures" / "eurosat_samples.pdf")
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    banner("Figure: EuroSAT samples per class")
    data = load_eurosat()
    fig = plot_sample_grid(data["images"], data["labels"], data["classes"],
                           per_class=args.per_class, seed=args.seed)
    with atomic_path(args.out) as tmp:
        fig.savefig(tmp, format=args.out.suffix.lstrip("."), dpi=300)
    plt.close(fig)
    write_meta(args.out, per_class=args.per_class, seed=args.seed)
    print(f"✓ written to {args.out}")


if __name__ == "__main__":
    main()
