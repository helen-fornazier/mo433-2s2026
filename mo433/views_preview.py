"""Step E5 check: preview of the pre-generated views, for visual inspection.

Draws one training image per class and its K views, generated exactly as step E6 will
(same transform, same per-(image, view) seeds), so the preview shows the real views.
Saved to $MO433_DATA/figures/views_preview.png, together with the effective transform
parameters in its .json.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mo433 import config
from mo433.dataset import load_eurosat
from mo433.io import atomic_path, banner, data_dir, load_pickle, write_meta
from mo433.transforms import describe, generate_view, make_view_transform, view_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="which images are shown")
    args = parser.parse_args()
    root = data_dir()
    out = root / "figures" / "views_preview.png"

    banner("E5: preview of the training views")
    data = load_eurosat()
    train_idx = load_pickle(root / "split_idx.pkl")["train"]
    y_train = data["labels"][train_idx]
    rng = np.random.default_rng(args.seed)
    positions = [int(rng.choice(np.flatnonzero(y_train == k))) for k in range(config.NUM_CLASSES)]

    transform = make_view_transform()
    fig, axes = plt.subplots(len(positions), config.K_VIEWS + 1,
                             figsize=((config.K_VIEWS + 1) * 1.1, len(positions) * 1.15))
    for row, pos in enumerate(positions):
        image = data["images"][train_idx[pos]]
        axes[row, 0].imshow(image, interpolation="nearest")
        axes[row, 0].set_ylabel(data["classes"][y_train[pos]], rotation=0, ha="right",
                                va="center", fontsize=8)
        for k in range(config.K_VIEWS):
            view = generate_view(image, view_seed(pos, k), transform)
            axes[row, k + 1].imshow(view, interpolation="nearest")
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].set_title("original", fontsize=8)
    for k in range(config.K_VIEWS):
        axes[0, k + 1].set_title(f"view {k}", fontsize=8)
    plt.suptitle(f"Training views (min_scale={config.MIN_SCALE}, blur sigmas={config.BLUR_SIGMAS})",
                 fontweight="bold")
    plt.tight_layout()
    with atomic_path(out) as tmp:
        fig.savefig(tmp, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    write_meta(out, training_positions=positions, transform=describe(transform))
    print(f"✓ written to {out}")


if __name__ == "__main__":
    main()
