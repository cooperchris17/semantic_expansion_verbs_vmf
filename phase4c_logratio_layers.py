# -*- coding: utf-8 -*-
"""
phase4c_logratio_layers.py
Reproduce the two-panel Fig. 4: raw kappa difference vs. anisotropy-aware
log-ratio across all BERT layers, for the A1 vs B1+ endpoints.

This is a lightweight plotting step: it reads the kappa table produced by
phase3_vmf.py (output/kappa_all_layers.csv) and, for each verb, computes at
every layer
    (a) raw difference   kappa(A1) - kappa(B1+)
    (b) log-ratio        ln(kappa(A1) / kappa(B1+))
The contrast makes the anisotropy point visual: the raw difference is inflated
at the shallow layers, whereas the scale-free log-ratio stays bounded and shows
where the genuine A1-B1+ concentration gap sits. No BERT run is needed.

Usage:
  python phase4c_logratio_layers.py --kappa_csv output/kappa_all_layers.csv \
      --verbs take make like
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# per-verb marker/colour, matching the paper figure
VERB_STYLE = {
    "take": ("o", "tab:blue"),
    "make": ("s", "tab:orange"),
    "like": ("^", "tab:green"),
}


def load_kappa(kappa_csv):
    """Read the long-form kappa table (columns: verb, level, layer, kappa, n)."""
    df = pd.read_csv(kappa_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def plot_raw_vs_logratio(df, verbs, out_png="output/fig_raw_vs_logratio.png"):
    """Two-panel layer-wise figure: raw difference (left) and log-ratio (right)."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

    for verb in verbs:
        style = VERB_STYLE.get(verb, ("o", None))
        marker, color = style
        p = (df[df["verb"] == verb]
             .pivot(index="layer", columns="level", values="kappa"))
        if "A1" not in p or "B1+" not in p:
            print(f"[warn] {verb}: A1 or B1+ missing, skipped.")
            continue
        raw = p["A1"] - p["B1+"]
        logr = np.log(p["A1"]) - np.log(p["B1+"])   # = ln(kappa_A1 / kappa_B1+)

        axes[0].plot(p.index, raw, marker=marker, color=color, linewidth=1.8,
                     markersize=6, label=verb)
        axes[1].plot(p.index, logr, marker=marker, color=color, linewidth=1.8,
                     markersize=6, label=verb)

        # annotate where the log-ratio peaks (e.g. make relocates to L11)
        if verb == "make":
            peak = logr.idxmax()
            axes[1].annotate(f"make peak L{peak}",
                             xy=(peak, logr.loc[peak]),
                             xytext=(peak + 0.3, logr.loc[peak] + 0.03),
                             fontsize=10, color=color)

    for ax in axes:
        ax.axhline(0, color="grey", linestyle=":", linewidth=0.9)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=11, frameon=False)
        ax.set_xticks(range(0, 13, 2))

    axes[0].set_title(r"(a) Raw difference  $\kappa_{A1}-\kappa_{B1+}$",
                      fontsize=13)
    axes[1].set_title(r"(b) Log-ratio  $\ln(\kappa_{A1}/\kappa_{B1+})$",
                      fontsize=13)
    axes[1].set_xlabel("BERT layer (0 = embeddings)", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa_csv", default="output/kappa_all_layers.csv",
                        help="kappa table written by phase3_vmf.py")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--out", default="output/fig_raw_vs_logratio.png")
    args = parser.parse_args()

    if not os.path.exists(args.kappa_csv):
        raise SystemExit(
            f"[error] {args.kappa_csv} not found. Run phase3_vmf.py first to "
            f"produce the kappa table.")

    df = load_kappa(args.kappa_csv)
    plot_raw_vs_logratio(df, args.verbs, out_png=args.out)


if __name__ == "__main__":
    main()
