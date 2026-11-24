"""Step E25: every results table and figure of the report, from $MO433_DATA only (spec 3.8).

    python -m mo433.report

Reads the probes, diagnostics, sanity checks, training logs, latency and student info, and
writes LaTeX tables to $REPORT/tables/ and PDF figures to $REPORT/figures/ (Makefile variable
REPORT). The report \\input{}s these tables and \\includegraphics these figures, with captions
and labels kept in main.tex (except the longtables of the appendix, which carry their own). The
few tables of the setup typed in main.tex come from other outputs, listed in the README.

Aggregation (report, sec:protocol): for an encoder group (row, width, unlabeled percentage)
and a label fraction, mean and sample standard deviation (ddof=1) over all (seed, draw)
pairs; at 100% labels there is one draw, so over seeds only. Row D has one instance: its s.d.
is over draws, and at 100% it is a single value. The C-B gap is the difference of the means,
its spread sqrt(sd_B^2 + sd_C^2), and it is "beyond the spread" when |gap| > spread
(tab:interpretation).
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mo433 import config
from mo433.io import REPO_DIR, atomic_path, banner, data_dir, load_pickle, write_meta
from mo433.train import main_runs

REPORT_DIR = REPO_DIR.parent / "relatorio"
PROBES = ("knn", "linear", "rbf")
PROBE_NAMES = {"knn": "k-NN", "linear": "Linear SVM", "rbf": "RBF SVM"}
FRACTIONS = config.LABEL_FRACTIONS
ROW_NAMES = {"A": "random", "B": "VICReg", "C": "distilled", "D": "DINO ViT-S/16"}
COLORS = {"A": "#8c8c8c", "B": "#1f77b4", "C": "#ff7f0e", "D": "#000000"}
WIDTH_COLORS = {16: "#1f77b4", 32: "#d62728"}       # loss curves only
ENCODER_NAME = re.compile(r"(?P<row>[A-D])(?:-c(?P<c>\d+))?(?:-d(?P<d>\d+))?(?:-s(?P<s>\d+))?")


# ---------------------------------------------------------------------------
# Encoders and data
# ---------------------------------------------------------------------------
def encoders() -> list:
    """The 26 compared encoder instances: row A, the 20 main runs of B and C, and D."""
    a = [f"A-c{c}-s{s}" for c in config.WIDTHS for s in config.SEEDS[c]]
    return a + main_runs("B") + main_runs("C") + ["D"]


def parse(name: str) -> dict:
    m = ENCODER_NAME.fullmatch(name)
    if not m:
        raise ValueError(f"unknown encoder {name!r}")
    return {"row": m["row"], "c": int(m["c"]) if m["c"] else None,
            "d": int(m["d"]) if m["d"] else None, "seed": int(m["s"]) if m["s"] else None}


def group_of(name: str) -> str:
    """Encoder group: A-c16, B-c16-d100, ..., D (the instance without its seed)."""
    return re.sub(r"-s\d+$", "", name)


def load_probes(root, names) -> pd.DataFrame:
    """One row per (encoder, fraction, draw, probe), accuracy in %."""
    rows = []
    for name in names:
        for r in json.loads((root / "probes" / f"{name}.json").read_text())["results"]:
            rows.append({"encoder": name, "group": group_of(name), **parse(name),
                         "fraction": r["fraction"], "draw": r["draw"], "probe": r["probe"],
                         "acc": 100 * r["accuracy"]})
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean, sample s.d. and count over the (seed, draw) values of each group, fraction, probe."""
    g = df.groupby(["group", "fraction", "probe"])["acc"]
    out = g.agg(mean="mean", sd=lambda v: v.std(ddof=1) if len(v) > 1 else np.nan, n="count")
    return out


def gap(agg: pd.DataFrame, c: int, d: int, fraction: float, probe: str) -> dict:
    b, cc = agg.loc[(f"B-c{c}-d{d}", fraction, probe)], agg.loc[(f"C-c{c}-d{d}", fraction, probe)]
    spread = float(np.hypot(b["sd"], cc["sd"]))
    value = float(cc["mean"] - b["mean"])
    return {"gap": value, "spread": spread, "beyond": abs(value) > spread}


def load_json_dir(root, sub: str, names) -> dict:
    return {n: json.loads((root / sub / f"{n}.json").read_text()) for n in names}


# ---------------------------------------------------------------------------
# LaTeX formatting
# ---------------------------------------------------------------------------
def pm(mean, sd, digits: int = 1, bold: bool = False) -> str:
    s = f"{mean:.{digits}f}" if np.isnan(sd) else f"{mean:.{digits}f}\\,$\\pm$\\,{sd:.{digits}f}"
    return f"\\textbf{{{s}}}" if bold else s


def signed(x, digits: int = 1) -> str:
    return f"${x:+.{digits}f}$"


def thousands(x, digits: int = 0) -> str:
    """1152 -> 1{,}152 (LaTeX thousands separator)."""
    return f"{x:,.{digits}f}".replace(",", "{,}")


def yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def group_label(group: str, with_name: bool = False) -> str:
    """LaTeX name of a group: \\rowB-c16 (the percentage is said by the table)."""
    p = parse(group + "-s0" if group != "D" else group)
    s = f"\\row{p['row']}" + (f"-c{p['c']}" if p["c"] else "")
    return s + (f" ({ROW_NAMES[p['row']]})" if with_name else "")


def frac_label(f: float) -> str:
    return f"{100 * f:g}\\%"


def write_tex(path: Path, body: str, comment: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        Path(tmp).write_text(f"% GENERATED by mo433/report.py from $MO433_DATA -- do not edit.\n"
                             f"% {comment}\n{body}")


def main_groups(c: int) -> list:
    return [f"A-c{c}", f"B-c{c}-d100", f"C-c{c}-d100"]


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table_main(agg) -> str:
    lines = ["\\begin{tabular}{@{}l l c c c c@{}}", "\\toprule",
             "Probe & Encoder & " + " & ".join(frac_label(f) for f in FRACTIONS) + " \\\\"]
    for probe in PROBES:
        lines += ["\\midrule", f"\\multirow{{7}}{{*}}{{{PROBE_NAMES[probe]}}}"]
        for c in config.WIDTHS:
            groups = main_groups(c)
            best = {f: max(agg.loc[(g, f, probe), "mean"] for g in groups) for f in FRACTIONS}
            for g in groups:
                cells = [pm(*agg.loc[(g, f, probe), ["mean", "sd"]],
                            bold=agg.loc[(g, f, probe), "mean"] == best[f]) for f in FRACTIONS]
                lines.append(f" & {group_label(g)} & " + " & ".join(cells) + " \\\\")
        cells = [pm(*agg.loc[("D", f, probe), ["mean", "sd"]]) for f in FRACTIONS]
        lines.append(" & \\rowD     & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def table_gap(agg) -> str:
    lines = ["\\begin{tabular}{@{}l c ccc ccc ccc@{}}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{{PROBE_NAMES[p]}}}" for p in PROBES)
             + " \\\\", "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}",
             "Size & Labels & " + " & ".join(["gap & $\\sigma$ & beyond?"] * 3) + " \\\\"]
    for c in config.WIDTHS:
        lines.append("\\midrule")
        for i, f in enumerate(FRACTIONS):
            first = f"\\multirow{{4}}{{*}}{{c{c}}}" if i == 0 else ""
            cells = []
            for p in PROBES:
                g = gap(agg, c, 100, f, p)
                cells += [signed(g["gap"]), f"{g['spread']:.1f}", yes_no(g["beyond"])]
            lines.append(f"{first} & {frac_label(f)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def table_ablation(agg) -> str:
    lines = ["\\begin{tabular}{@{}c ccc ccc c@{}}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{c{c}}}" for c in config.WIDTHS)
             + " & Same \\\\", "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Labels & " + " & ".join(["50\\% data & 100\\% data & $\\Delta$"] * 2)
             + " & direction? \\\\", "\\midrule"]
    for f in FRACTIONS:
        cells, deltas = [], []
        for c in config.WIDTHS:
            g50, g100 = gap(agg, c, 50, f, "linear"), gap(agg, c, 100, f, "linear")
            deltas.append(g50["gap"] - g100["gap"])
            cells += [f"{signed(g50['gap'])}\\,$\\pm$\\,{g50['spread']:.1f}",
                      f"{signed(g100['gap'])}\\,$\\pm$\\,{g100['spread']:.1f}", signed(deltas[-1])]
        lines.append(f"{frac_label(f)} & " + " & ".join(cells)
                     + f" & {yes_no(np.sign(deltas[0]) == np.sign(deltas[1]))} \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def table_ablation_abs(agg) -> str:
    lines = ["\\begin{tabular}{@{}l c c c c@{}}", "\\toprule",
             "Encoder & " + " & ".join(frac_label(f) for f in FRACTIONS) + " \\\\", "\\midrule"]
    for c in config.WIDTHS:
        for row in "BC":
            g = f"{row}-c{c}-d50"
            cells = [pm(*agg.loc[(g, f, "linear"), ["mean", "sd"]]) for f in FRACTIONS]
            lines.append(f"\\row{row}-c{c} (50\\% data) & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def seed_stats(values: list, digits: int) -> str:
    v = np.asarray(values, dtype=float)
    return pm(v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan, digits)


def table_diag(agg, diag: dict) -> str:
    lines = ["\\begin{tabular}{@{}l c c c c@{}}", "\\toprule",
             "Encoder & NMI & ARI & Lin.\\ SVM 100\\% (\\%) & Lin.\\ SVM 1\\% (\\%) \\\\", "\\midrule"]
    for g in main_groups(16) + main_groups(32) + ["D"]:
        names = [n for n in diag if group_of(n) == g]
        cells = [seed_stats([diag[n][k] for n in names], 3) for k in ("kmeans_nmi", "kmeans_ari")]
        cells += [pm(*agg.loc[(g, f, "linear"), ["mean", "sd"]]) for f in (1.0, 0.01)]
        lines.append(f"{group_label(g)} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def table_sanity(sanity: dict) -> str:
    def cells(names: list) -> str:
        return " & ".join(seed_stats([scale * sanity[n][key] for n in names], digits)
                          for key, digits, scale in (("knn_100", 1, 100), ("embedding_std", 3, 1),
                                                     ("kmeans_nmi", 3, 1)))
    lines = ["\\begin{tabular}{@{}l c c c@{}}", "\\toprule",
             "Encoder & k-NN acc.\\ (\\%) & Emb.\\ s.d. & NMI \\\\", "\\midrule",
             f"\\rowD\\ (teacher) & {cells(['D'])} \\\\"]
    for row in "ABC":
        for c in config.WIDTHS:
            g = f"{row}-c{c}" + ("" if row == "A" else "-d100")
            lines.append(f"{group_label(g)} & {cells([n for n in sanity if group_of(n) == g])} \\\\")
    lines += ["\\midrule", "\\multicolumn{4}{@{}l}{\\emph{Negative controls}} \\\\"]
    for name, label in (("pixels", "Raw pixels (12{,}288-d)"), ("D-random", "ViT-S/16, random weights")):
        lines.append(f"{label} & {cells([name])} \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def table_encoders(info: dict, latency: dict) -> str:
    lat = latency["encoders"]
    lines = ["\\begin{tabular}{@{}l l l c r r r r@{}}", "\\toprule",
             "Row & Encoder & Training & Input & Params & Feat.\\ dim & GFLOPs & Latency \\\\",
             "    &         &          & (px)  & (M)    &            & /img   & (ms/batch) \\\\",
             "\\midrule"]
    training = {"A": "none (random init)", "B": "VICReg from scratch", "C": "distilled from \\rowD"}
    for row in "ABC":
        for c in config.WIDTHS:
            s = info[f"c{c}"]
            lines.append(f"\\row{row} & Student $c{{=}}{c}$ & {training[row]} & {config.IMAGE_SIZE} & "
                         f"{s['params'] / 1e6:.2f} & {s['feature_dim']} & "
                         f"{s['flops_per_image'] / 1e9:.2f} & {thousands(lat[f'c{c}']['ms_per_batch'])} \\\\")
    t = info["teacher"]
    lines.append(f"\\rowD & DINO ViT-S/16 & frozen public checkpoint & {config.TEACHER_RES} & "
                 f"{t['params'] / 1e6:.1f} & {config.TEACHER_DIM} & {t['flops_per_image'] / 1e9:.2f} & "
                 f"{thousands(lat['teacher']['ms_per_batch'])} \\\\")
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


def run_logs(root, names) -> dict:
    return {n: json.loads((root / "runs" / f"{n}_log.json").read_text()) for n in names}


def table_cost(agg, logs: dict, cache_gpu_hours: float) -> str:
    c, f = config.COST_WIDTH, config.COST_FRACTION
    n_c_runs = len(main_runs("C"))
    share = cache_gpu_hours / n_c_runs
    train = {row: float(np.mean([l["gpu_hours"] for n, l in logs.items()
                                 if group_of(n) == f"{row}-c{c}-d100"])) for row in "BC"}
    excl = {"B": train["B"], "C": train["C"] + share}
    incl = {"B": train["B"], "C": excl["C"] + config.TEACHER_PRETRAIN_GPU_HOURS}
    a = agg.loc[(f"A-c{c}", f, "linear"), "mean"]
    gain = {row: agg.loc[(f"{row}-c{c}-d100", f, "linear"), "mean"] - a for row in "BC"}
    extra = gain["C"] - gain["B"]

    def per_point(cost):
        d = cost["C"] - cost["B"]
        return "none (\\rowC\\ cheaper)" if d <= 0 else thousands(d / extra)
    rows = [(f"Teacher pretraining, {config.TEACHER_PRETRAIN_EPOCHS} epochs (extrapolated)", "--",
             thousands(config.TEACHER_PRETRAIN_GPU_HOURS)),
            ("\\quad published 300-epoch run (lower bound)", "--",
             thousands(config.TEACHER_PRETRAIN_GPU_HOURS_300)),
            ("Teacher feature cache (once, shared by all runs)", "--", f"{cache_gpu_hours:.2f}"),
            (f"\\quad share per run (cache / {n_c_runs} \\rowC\\ runs)", "--", f"{share:.3f}"),
            ("Student training, per run (mean over seeds)", f"{train['B']:.2f}", f"{train['C']:.2f}"),
            None,
            ("Total per run, excluding teacher pretraining", f"{excl['B']:.2f}", f"{excl['C']:.2f}"),
            ("Total per run, including teacher pretraining", f"{incl['B']:.2f}", thousands(incl['C'])),
            None,
            ("Accuracy gain over \\rowA\\ (points)", f"{gain['B']:.1f}", f"{gain['C']:.1f}"),
            ("Gain per GPU-hour, excluding teacher pretraining",
             f"{gain['B'] / excl['B']:.1f}", f"{gain['C'] / excl['C']:.1f}"),
            ("Gain per GPU-hour, including teacher pretraining",
             f"{gain['B'] / incl['B']:.1f}", f"{gain['C'] / incl['C']:.3f}")]
    lines = ["\\begin{tabular}{@{}l r r@{}}", "\\toprule", "Item & \\rowB & \\rowC \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r is None else " & ".join(r) + " \\\\")
    lines += ["\\midrule",
              f"Extra GPU-h of \\rowC\\ per extra point, excl.\\ pretraining & "
              f"\\multicolumn{{2}}{{r}}{{{per_point(excl)}}} \\\\",
              f"Extra GPU-h of \\rowC\\ per extra point, incl.\\ pretraining & "
              f"\\multicolumn{{2}}{{r}}{{{per_point(incl)}}} \\\\"]
    return "\n".join(lines + ["\\bottomrule", "\\end{tabular}", ""])


LONG_HEAD = """\\toprule
{head} \\\\
\\midrule
\\endfirsthead
\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\small\\itshape Table~\\ref{{{label}}} (continued)}}\\\\
\\toprule
{head} \\\\
\\midrule
\\endhead
\\bottomrule
\\endfoot"""


def longtable(spec: str, caption: str, label: str, head: str, ncol: int, body: list) -> str:
    return "\n".join([f"\\begin{{longtable}}{{{spec}}}", f"\\caption{{{caption}}}",
                      f"\\label{{{label}}}\\\\",
                      LONG_HEAD.format(head=head, ncol=ncol, label=label), *body,
                      "\\end{longtable}", ""])


def table_results_long(agg, info, latency) -> str:
    groups = [g for c in config.WIDTHS for g in main_groups(c)] + \
             [f"{r}-c{c}-d50" for c in config.WIDTHS for r in "BC"] + ["D"]
    body = []
    for g in groups:
        p = parse(g + "-s0" if g != "D" else g)
        key = f"c{p['c']}" if p["c"] else "teacher"
        params = info[key]["params"] / 1e6
        ms = latency["encoders"][key]["ms_per_batch"]
        name = g.replace("-d100", "").replace("-d50", " (50\\% data)") if g != "D" else "D"
        name += f" ({ROW_NAMES[p['row']]})"
        for f in FRACTIONS:
            for probe in PROBES:
                body.append(f"{name} & {params:.2f} & {thousands(ms)} & {frac_label(f)} & "
                            f"{PROBE_NAMES[probe]} & {pm(*agg.loc[(g, f, probe), ['mean', 'sd']])} \\\\")
            body.append("\\addlinespace[2pt]")
    caption = ("Full results table (Section~3.8 of the project specification): one row per "
               "(encoder group $\\times$ label fraction $\\times$ probe). Accuracy in \\%, mean "
               "$\\pm$ s.d.\\ over seeds and labeled-subset draws (A, B, C; at 100\\% labels over "
               "seeds only); D is a fixed checkpoint (s.d.\\ over draws only). Students without a "
               "data label use 100\\% of the unlabeled images. Params in millions; latency in ms "
               f"per batch of {latency['batch']} images on the laptop CPU (\\cref{{tab:encoders}}).")
    return longtable("@{}l r r c l c@{}", caption, "tab:results-long",
                     "Encoder & Params (M) & Latency (ms) & Labels & Probe & Accuracy (\\%)", 6, body)


def table_per_run(df) -> str:
    body, last = [], None
    for name in encoders():
        if last is not None and group_of(name) != last:
            body.append("\\addlinespace[2pt]")
        last = group_of(name)
        lin = df[(df.encoder == name) & (df.probe == "linear")]
        cells = []
        for f in FRACTIONS:
            v = lin[lin.fraction == f]["acc"].to_numpy()
            cells.append(pm(v.mean(), v.std(ddof=1) if len(v) > 1 else np.nan))
        body.append(f"\\texttt{{{name}}} & " + " & ".join(cells) + " \\\\")
    caption = ("Linear-SVM accuracy (\\%) of every individual encoder instance (26 encoders). At "
               "1/10/50\\% labels: mean $\\pm$ s.d.\\ over the three labeled-subset draws; at "
               "100\\%: single value. Required in particular for the larger student, which has "
               "only two seeds.")
    return longtable("@{}l c c c c@{}", caption, "tab:per-run",
                     "Encoder instance & " + " & ".join(frac_label(f) for f in FRACTIONS), 5, body)


def table_collapse_per_run(checks: list, logs: dict) -> str:
    body, last = [], None
    for r in checks:
        if last is not None and group_of(r["name"]) != last:
            body.append("\\addlinespace[2pt]")
        last = group_of(r["name"])
        loss = logs[r["name"]]["epochs"][-1]["loss"]
        verdict = "ok" if not r["problems"] else ", ".join(r["problems"])
        body.append(f"\\texttt{{{r['name']}}} & {loss:.3f} & {100 * r['knn_100']:.1f} & "
                    f"{r['embedding_std']:.3f} & {r['kmeans_nmi']:.3f} & {verdict} \\\\")
    caption = ("Collapse check of every trained encoder (20 runs): final training loss, k-NN "
               "accuracy at 100\\% labels, mean per-dimension s.d.\\ of the L2-normalized test "
               "embeddings, and k-means NMI on the test split. Verdict by the criterion fixed "
               "before the main runs (\\cref{sec:res-sanity}).")
    return longtable("@{}l c c c c c@{}", caption, "tab:collapse-per-run",
                     "Run & Final loss & k-NN acc.\\ (\\%) & Emb.\\ s.d. & NMI & Verdict", 6, body)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp, open(tmp, "wb") as f:
        fig.savefig(f, format="pdf", bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def style():
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False})


def label_efficiency_lines(ax, agg, probe: str, c: int):
    x = [100 * f for f in FRACTIONS]
    for g in main_groups(c) + ["D"]:
        row = g[0]
        m = np.array([agg.loc[(g, f, probe), "mean"] for f in FRACTIONS])
        s = np.nan_to_num(np.array([agg.loc[(g, f, probe), "sd"] for f in FRACTIONS]))
        ax.plot(x, m, ":" if row == "D" else "-", marker="o", ms=3, color=COLORS[row], lw=1.3,
                label=f"{row} ({ROW_NAMES[row]})")
        ax.fill_between(x, m - s, m + s, color=COLORS[row], alpha=0.18, lw=0)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{v:g}" for v in x])
    ax.minorticks_off()
    ax.grid(alpha=0.3, lw=0.5)


def y_range(agg, groups, probe: str, pad: float = 1.5) -> tuple:
    rows = [agg.loc[(g, f, probe)] for g in groups for f in FRACTIONS]
    lo = min(r["mean"] - np.nan_to_num(r["sd"]) for r in rows)
    hi = max(r["mean"] + np.nan_to_num(r["sd"]) for r in rows)
    return lo - pad, hi + pad


def fig_label_efficiency(agg, probes=("linear",)) -> plt.Figure:
    """One column per student width; the y axis is broken between row A (bottom) and rows
    B, C, D (top), which are 20-50 points apart, so that the top lines stay readable."""
    fig = plt.figure(figsize=(6.4, 2.9 * len(probes)))
    outer = fig.add_gridspec(len(probes), len(config.WIDTHS), hspace=0.45, wspace=0.12)
    trained = [g for c in config.WIDTHS for g in main_groups(c)[1:]] + ["D"]
    for i, probe in enumerate(probes):
        top_lim = y_range(agg, trained, probe)
        bottom_lim = y_range(agg, [f"A-c{c}" for c in config.WIDTHS], probe)
        for j, c in enumerate(config.WIDTHS):
            inner = outer[i, j].subgridspec(2, 1, height_ratios=(2.2, 1), hspace=0.08)
            top, bottom = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            for ax in (top, bottom):
                label_efficiency_lines(ax, agg, probe, c)
            top.set_ylim(*top_lim)
            bottom.set_ylim(*bottom_lim)
            top.spines["bottom"].set_visible(False)
            top.tick_params(bottom=False, labelbottom=False)
            for ax, y in ((top, 0), (bottom, 1)):    # the break marks on the left axis
                ax.plot([-0.015, 0.015], [y - 0.02, y + 0.02], transform=ax.transAxes,
                        color="k", lw=0.8, clip_on=False)
            bottom.set_xlabel("labeled fraction (%)")
            top.set_title(f"{PROBE_NAMES[probe]}, student c{c}" if len(probes) > 1
                          else f"student c{c}")
            if j == 0:
                top.set_ylabel(f"{PROBE_NAMES[probe]} test accuracy (%)", y=0.3)
            else:
                top.tick_params(labelleft=False)
                bottom.tick_params(labelleft=False)
            if i == 0 and j == len(config.WIDTHS) - 1:
                handles, labels = top.get_legend_handles_labels()
                top.legend(handles, labels, loc="lower right", frameon=False)
    return fig


def fig_gap_params(agg, info) -> plt.Figure:
    params = [info[f"c{c}"]["params"] / 1e6 for c in config.WIDTHS]
    fig, axes = plt.subplots(1, len(config.UNLAB_FRACTIONS), figsize=(6.4, 2.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, d in zip(axes, sorted(config.UNLAB_FRACTIONS, reverse=True)):
        for k, f in enumerate(FRACTIONS):
            gs = [gap(agg, c, d, f, "linear") for c in config.WIDTHS]
            offset = 1 + 0.04 * (k - 1.5)          # spread the points sideways to show the bars
            ax.errorbar([p * offset for p in params], [g["gap"] for g in gs],
                        yerr=[g["spread"] for g in gs], marker="o", ms=3, capsize=2, lw=1.2,
                        color=cmap(k / (len(FRACTIONS) - 1) * 0.85), label=f"{100 * f:g}% labels")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(params, [f"{p:.2f}M\n(c{c})" for p, c in zip(params, config.WIDTHS)])
        ax.minorticks_off()
        ax.set_xlim(params[0] / 1.6, params[-1] * 1.6)
        ax.set_xlabel("student parameters")
        ax.set_title(f"{d}% of the unlabeled data")
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("C $-$ B gap, linear SVM (points)")
    axes[-1].legend(frameon=False, loc="best")
    fig.tight_layout()
    return fig


def fig_loss(logs: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.5))
    titles = {"B": "B: VICReg loss", "C": "C: distillation loss (1 $-$ cos)"}
    for ax, row in zip(axes, "BC"):
        for name, log in logs.items():
            p = parse(name)
            if p["row"] != row:
                continue
            ep = [e["epoch"] for e in log["epochs"]]
            ax.plot(ep, [e["loss"] for e in log["epochs"]], color=WIDTH_COLORS[p["c"]],
                    ls="-" if p["d"] == 100 else "--", lw=0.8, alpha=0.8)
        ax.set_title(titles[row])
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, lw=0.5)
        if row == "C":
            ax.set_yscale("log")
    axes[0].set_ylabel("training loss")
    handles = [plt.Line2D([], [], color=WIDTH_COLORS[c], label=f"c{c}") for c in config.WIDTHS]
    handles += [plt.Line2D([], [], color="gray", label="100% data"),
                plt.Line2D([], [], color="gray", ls="--", label="50% data")]
    axes[1].legend(handles=handles, frameon=False)
    fig.tight_layout()
    return fig


def fig_projections(root, names: list, titles: list, classes: list, y_test) -> plt.Figure:
    fig, axes = plt.subplots(1, len(names), figsize=(6.6, 1.95))
    cmap = plt.get_cmap("tab10")
    for ax, name, title in zip(axes, names, titles):
        proj = np.load(root / "diag" / f"{name}_proj.npy")
        ax.scatter(proj[:, 0], proj[:, 1], c=[cmap(k) for k in y_test], s=0.6, lw=0,
                   rasterized=True)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.5)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4, color=cmap(k), label=cls)
               for k, cls in enumerate(classes)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    return fig


def confusion(root, name: str, y_train, y_test, idx) -> np.ndarray:
    """Row-normalized confusion matrix of the linear SVM fitted on the training rows ``idx``."""
    from sklearn.metrics import confusion_matrix

    from mo433.evaluate import fit_preprocessor, linear_svm
    tr = np.load(root / "feats" / f"{name}_train.npy")
    te = np.load(root / "feats" / f"{name}_test.npy")
    pre = fit_preprocessor(tr)
    clf = linear_svm(config.SVM_SELECTED["linear"]["C"]).fit(pre.transform(tr)[idx], y_train[idx])
    return confusion_matrix(y_test, clf.predict(pre.transform(te)), normalize="true")


def fig_confusion(mats: dict, classes: list) -> plt.Figure:
    fig, axes = plt.subplots(1, len(mats), figsize=(6.6, 3.2))
    short = [c.replace("HerbaceousVegetation", "HerbVeg").replace("PermanentCrop", "PermCrop")
             .replace("AnnualCrop", "AnnCrop") for c in classes]
    for ax, (title, m) in zip(axes, mats.items()):
        ax.imshow(m, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(m)):
            for j in range(len(m)):
                if m[i, j] >= 0.05:
                    ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=5,
                            color="white" if m[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(m)), short, rotation=90)
        ax.set_yticks(range(len(m)), short)
        ax.set_xlabel("predicted")
        ax.set_title(title)
    axes[0].set_ylabel("true class")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPORT_DIR), help="the report directory")
    args = parser.parse_args()
    root, out = data_dir(), Path(args.out)
    tables, figures = out / "tables", out / "figures"
    banner("E25: tables and figures of the report")
    style()

    names = encoders()
    df = load_probes(root, names)
    agg = aggregate(df)
    info = json.loads((root / "student_info.json").read_text())
    latency = json.loads((root / "latency.json").read_text())
    sanity = load_json_dir(root, "sanity", names + ["pixels", "D-random"])
    diag = load_json_dir(root, "diag", names)
    trained = main_runs("B") + main_runs("C")
    logs = run_logs(root, trained)
    cache = json.loads((root / f"teacher_feats_K{config.K_VIEWS}.npy.json").read_text())["gpu_hours"]
    from mo433.check_runs import check_all
    checks = check_all(root)

    outputs = {"main": table_main(agg), "gap": table_gap(agg), "ablation": table_ablation(agg),
               "ablation_abs": table_ablation_abs(agg), "diag": table_diag(agg, diag),
               "sanity": table_sanity(sanity), "encoders": table_encoders(info, latency),
               "cost": table_cost(agg, logs, cache),
               "results_long": table_results_long(agg, info, latency),
               "results_per_run": table_per_run(df),
               "collapse_per_run": table_collapse_per_run(checks, logs)}
    for name, body in outputs.items():
        write_tex(tables / f"{name}.tex", body, f"table {name}")

    labels = np.load(root / "eurosat" / "labels.npy")
    split = load_pickle(root / "split_idx.pkl")
    y_train, y_test = labels[split["train"]], labels[split["test"]]
    classes = json.loads((root / "eurosat" / "index.json").read_text())["classes"]
    subset = np.asarray(load_pickle(root / "label_subsets.pkl")[(config.CONFUSION_FRACTION, 0)])
    figs = {
        "label_efficiency": fig_label_efficiency(agg),
        "label_eff_others": fig_label_efficiency(agg, ("knn", "rbf")),
        "gap_vs_params": fig_gap_params(agg, info),
        "loss_curves": fig_loss(logs),
        "proj_c16": fig_projections(root, ["A-c16-s0", "B-c16-d100-s0", "C-c16-d100-s0", "D"],
                                    ["(a) A, random", "(b) B, VICReg", "(c) C, distilled",
                                     "(d) D, teacher"], classes, y_test),
        "proj_c32": fig_projections(root, ["A-c32-s0", "B-c32-d100-s0", "C-c32-d100-s0", "D"],
                                    ["(a) A-c32, random", "(b) B-c32, VICReg",
                                     "(c) C-c32, distilled", "(d) D, teacher"], classes, y_test),
        "confusion": fig_confusion({f"{n} ({frac_label(config.CONFUSION_FRACTION).replace(chr(92), '')} labels)":
                                    confusion(root, n, y_train, y_test, subset)
                                    for n in ("B-c16-d100-s0", "C-c16-d100-s0")}, classes),
    }
    for name, fig in figs.items():
        save_fig(fig, figures / f"{name}.pdf")
    write_meta(tables / "main.tex", tables=sorted(outputs), figures=sorted(figs))
    print(f"✓ {len(outputs)} tables in {tables}, {len(figs)} figures in {figures}")


if __name__ == "__main__":
    main()
