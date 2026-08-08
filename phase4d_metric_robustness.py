# -*- coding: utf-8 -*-
"""
phase4d_metric_robustness.py
Reproduce Fig. 5 (Section 5.6): vMF kappa against mean pairwise cosine distance.

This is a metric-robustness check. If the two dispersion measures track each
other, the paper's conclusions do not hinge on which one is used. The figure
scatters, per verb, every (layer, level) point as kappa (x) vs. mean pairwise
cosine distance (y); the per-verb Pearson correlation r is shown in the legend
and is strongly negative (higher kappa = more concentrated = smaller distance).

Like phase4c, this is a lightweight plotting step: it reads the table written by
phase4b (output/dispersion_all.csv) and does not run BERT.

Usage:
  python phase4d_metric_robustness.py --disp_csv output/dispersion_all.csv \
      --verbs take make like
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt

# per-verb marker/colour, matching the other figures
VERB_STYLE = {
    "take": ("o", "tab:blue"),
    "make": ("s", "tab:orange"),
    "like": ("^", "tab:green"),
}


def load_dispersion(disp_csv):
    """Read the dispersion table (columns: verb, level, layer, kappa, mean_dist, n)."""
    df = pd.read_csv(disp_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def plot_metric_robustness(df, verbs, out_png="output/fig_metric_robustness.png"):
    """Scatter kappa vs. mean pairwise cosine distance, one series per verb,
    with the per-verb Pearson r in the legend."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    for verb in verbs:
        s = df[df["verb"] == verb].dropna(subset=["kappa", "mean_dist"])
        if s.empty:
            print(f"[warn] {verb}: no data, skipped.")
            continue
        marker, color = VERB_STYLE.get(verb, ("o", None))
        r = s["kappa"].corr(s["mean_dist"])   # Pearson correlation
        ax.scatter(s["kappa"], s["mean_dist"], marker=marker, color=color,
                   alpha=0.75, edgecolors="none", s=45,
                   label=f"{verb} (r={r:.2f})")

    ax.set_xlabel(r"vMF $\kappa$", fontsize=12)
    ax.set_ylabel("mean pairwise cosine distance", fontsize=12)
    ax.set_title("Metric robustness: concentration vs. dispersion", fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=11, frameon=False)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disp_csv", default="output/dispersion_all.csv",
                        help="dispersion table written by phase4b")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--out", default="output/fig_metric_robustness.png")
    args = parser.parse_args()

    if not os.path.exists(args.disp_csv):
        raise SystemExit(
            f"[error] {args.disp_csv} not found. Run phase4b_dispersion_lines.py "
            f"first to produce the dispersion table.")

    df = load_dispersion(args.disp_csv)
    plot_metric_robustness(df, args.verbs, out_png=args.out)


if __name__ == "__main__":
    main()
