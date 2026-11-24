"""Step E27: compare a from-scratch reproduction with the original results (spec 3.8).

    python -m mo433.repro_compare --ref ../data          ($MO433_DATA = the reproduction)

Aggregates the probes of both data directories exactly as the report does (mo433.report) and
compares every cell of tab:main (encoder group x label fraction x probe, mean over seeds and
draws). Only the encoders present in both directories are used, so a partial reproduction (for
example one seed per row) is compared on what it has. A difference is put in perspective with
the original spread (the s.d. over seeds and draws of that cell): cuDNN is not bitwise
deterministic, so a rerun with the same seed is a slightly different run, and differences of the
order of the seed-to-seed spread are expected.

Writes $MO433_DATA/repro_compare.json: per cell the two means, the difference and the original
spread; and the largest and mean absolute differences.
"""

import argparse
from pathlib import Path

import numpy as np

from mo433 import config
from mo433.io import banner, data_dir, save_json, write_meta
from mo433.report import aggregate, encoders, load_probes, main_groups


def available(root: Path) -> list:
    return [n for n in encoders() if (root / "probes" / f"{n}.json").exists()]


def compare(ref: Path, new: Path) -> dict:
    names = [n for n in available(new) if n in set(available(ref))]
    a, b = aggregate(load_probes(ref, names)), aggregate(load_probes(new, names))
    groups = [g for c in config.WIDTHS for g in main_groups(c)] + ["D"]
    cells = []
    for key in b.index:
        if key[0] not in groups or key not in a.index:
            continue
        diff = float(b.loc[key, "mean"] - a.loc[key, "mean"])
        cells.append({"group": key[0], "fraction": key[1], "probe": key[2],
                      "original": float(a.loc[key, "mean"]), "reproduced": float(b.loc[key, "mean"]),
                      "diff": diff, "original_sd": float(a.loc[key, "sd"]),
                      "within_sd": bool(abs(diff) <= np.nan_to_num(a.loc[key, "sd"]))})
    diffs = np.abs([c["diff"] for c in cells])
    return {"encoders": names, "n_cells": len(cells),
            "max_abs_diff": float(diffs.max()) if len(cells) else None,
            "mean_abs_diff": float(diffs.mean()) if len(cells) else None,
            "cells_within_sd": int(sum(c["within_sd"] for c in cells)), "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", required=True, help="the data directory of the original results")
    args = parser.parse_args()
    new, ref = data_dir(), Path(args.ref).resolve()
    if new.resolve() == ref:
        raise SystemExit("the reproduction and the reference are the same directory")
    banner(f"E27: reproduction {new} vs. original {ref}")
    out = new / "repro_compare.json"
    result = compare(ref, new)
    if not result["n_cells"]:
        raise SystemExit("no encoder has probes in both directories")
    worst = sorted(result["cells"], key=lambda c: -abs(c["diff"]))[:5]
    for c in worst:
        print(f"  {c['group']:<11} {100 * c['fraction']:>4g}% {c['probe']:<6} "
              f"{c['original']:5.1f} -> {c['reproduced']:5.1f}  ({c['diff']:+.2f}, "
              f"original s.d. {c['original_sd']:.2f})")
    save_json(result, out)
    write_meta(out, ref=str(ref), encoders=result["encoders"])
    print(f"✓ {len(result['encoders'])} encoders, {result['n_cells']} cells of tab:main: "
          f"max |diff| {result['max_abs_diff']:.2f}, mean {result['mean_abs_diff']:.2f} points, "
          f"{result['cells_within_sd']} within the original s.d.; written to {out}")


if __name__ == "__main__":
    main()
