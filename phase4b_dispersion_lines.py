# -*- coding: utf-8 -*-
"""
phase4b_dispersion_lines.py
Show semantic dispersion of verb vectors as layer-wise line plots, without any
dimensionality reduction (t-SNE/PCA collapse the 768-dim spread). Two metrics
are plotted so the conclusion is robust to metric choice:
  1. vMF kappa            : lower = more dispersed.
  2. mean pairwise cosine distance : higher = more dispersed.
The A1-B1+ gap curve shows at which layer the dispersion gap is largest.

Usage:
  python phase4b_dispersion_lines.py --csv JEFLL_ten_verbs.csv --verbs take make like
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shared_utils import (
    LEVEL_ORDER, load_data_with_ids, extract_verb_contexts,
    get_all_layer_vectors, load_bert, load_spacy,
    VonMisesFisherUtils,
)

TARGET_LEVELS = ["A1", "A2", "B1+"]
LEVEL_COLORS = {"A1": "#1f77b4", "A2": "#ff7f0e", "B1+": "#2ca02c"}
MAX_PER_LEVEL = 400   # cap per level; pairwise distance is O(n^2)


def collect_all_layers(csv_path, verb, tokenizer, model, device, nlp):
    """Return {level: ndarray(n, n_layers, 768)} for one verb."""
    data = load_data_with_ids(csv_path)
    if verb not in data:
        return {}
    out = {}
    for level in TARGET_LEVELS:
        sents = data[verb].get(level, [])
        if not sents:
            continue
        contexts = extract_verb_contexts(sents, verb, nlp, require_noun=False)
        if not contexts:
            continue
        v = get_all_layer_vectors(contexts, tokenizer, model, device)
        if v.shape[0] == 0:
            continue
        if v.shape[0] > MAX_PER_LEVEL:
            idx = np.random.RandomState(42).choice(
                v.shape[0], MAX_PER_LEVEL, replace=False)
            v = v[idx]
        out[level] = v
        print(f"  {verb} {level}: {v.shape[0]} vectors")
    return out


def mean_pairwise_cosine_distance(vecs):
    """Mean pairwise cosine distance (1 - similarity). Vectors are unit-length,
    so similarity = dot product. Computed via the Gram-sum identity to avoid the
    O(n^2) loop; the i==j diagonal is removed."""
    n = vecs.shape[0]
    if n < 2:
        return np.nan
    sum_vec = vecs.sum(axis=0)
    total_sim = sum_vec @ sum_vec                     # sum_i sum_j v_i.v_j (with diagonal)
    diag = n                                          # diagonal terms v_i.v_i = 1
    offdiag_sim = (total_sim - diag) / (n * (n - 1))  # mean similarity over i != j
    return 1.0 - offdiag_sim


def compute_dispersion(vecs_by_level):
    """Compute kappa and mean pairwise distance per level x layer.
    Returns long-form DataFrame(level, layer, kappa, mean_dist, n)."""
    rows = []
    for level, vecs in vecs_by_level.items():
        n_layers = vecs.shape[1]
        for layer in range(n_layers):
            lv = vecs[:, layer, :]
            _, kappa = VonMisesFisherUtils.estimate_vmf_parameters(lv)
            dist = mean_pairwise_cosine_distance(lv)
            rows.append({
                "level": level, "layer": layer,
                "kappa": kappa, "mean_dist": dist, "n": lv.shape[0],
            })
    return pd.DataFrame(rows)


def plot_dispersion_lines(disp_df, verb, peak_layer=None, out_png=None):
    """Left: kappa by layer. Right: mean pairwise distance by layer.
    Coloured by level; peak_layer (if given) marked with a vertical line."""
    if out_png is None:
        out_png = f"output/dispersion_{verb}.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    layers = sorted(disp_df["layer"].unique())

    for level in TARGET_LEVELS:
        sub = disp_df[disp_df["level"] == level].sort_values("layer")
        if sub.empty:
            continue
        c = LEVEL_COLORS[level]
        ax1.plot(sub["layer"], sub["kappa"], marker="o", linewidth=2.3,
                 color=c, label=level)
        ax2.plot(sub["layer"], sub["mean_dist"], marker="o", linewidth=2.3,
                 color=c, label=level)

    for ax in (ax1, ax2):
        if peak_layer is not None:
            ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.5)
            ax.annotate(f"probing peak (L{peak_layer})",
                        xy=(peak_layer, ax.get_ylim()[1]),
                        xytext=(5, -15), textcoords="offset points",
                        fontsize=9, color="red")
        ax.set_xlabel("BERT Layer (0=embeddings, 12=final)", fontsize=12)
        ax.set_xticks(layers)
        ax.legend(title="CEFR", fontsize=10)
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel("vMF kappa", fontsize=12)
    ax1.set_title("kappa by layer (lower = more dispersed)", fontsize=12,
                  fontweight="bold")
    ax2.set_ylabel("Mean pairwise cosine distance", fontsize=12)
    ax2.set_title("Pairwise distance by layer (higher = more dispersed)",
                  fontsize=12, fontweight="bold")

    fig.suptitle(f"Layer-wise semantic dispersion: '{verb}'",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def plot_gap_curve(disp_df, verb, peak_layer=None, out_png=None):
    """Plot the A1-B1+ dispersion gap per layer as kappa(A1) - kappa(B1+).
    A larger value means a bigger A1-B1+ difference at that layer."""
    if out_png is None:
        out_png = f"output/gap_{verb}.png"
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    pivot = disp_df.pivot(index="layer", columns="level", values="kappa")
    if "A1" not in pivot or "B1+" not in pivot:
        print("[warn] A1 or B1+ missing; cannot draw gap curve")
        return
    gap = pivot["A1"] - pivot["B1+"]   # positive => A1 more concentrated (B1+ dispersed)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(gap.index, gap.values, marker="o", linewidth=2.8, color="#9467bd")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.6)
    if peak_layer is not None:
        ax.axvline(peak_layer, color="red", linestyle="--", alpha=0.5,
                   label=f"probing peak (L{peak_layer})")
        ax.legend(fontsize=10)
    best_layer = gap.idxmax()
    ax.scatter([best_layer], [gap.max()], s=160, facecolors="none",
               edgecolors="#9467bd", linewidths=2.5)
    ax.annotate(f"max gap at L{best_layer}",
                xy=(best_layer, gap.max()),
                xytext=(8, 8), textcoords="offset points",
                fontsize=10, fontweight="bold")

    ax.set_xlabel("BERT Layer", fontsize=12)
    ax.set_ylabel("kappa(A1) - kappa(B1+)\n(higher = bigger A1-B1+ dispersion gap)",
                  fontsize=11)
    ax.set_title(f"Where is the A1-B1+ dispersion gap largest? '{verb}'",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(sorted(disp_df["layer"].unique()))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="JEFLL_ten_verbs.csv")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--peak_layers", nargs="*",
                        default=["take=6", "make=11", "like=11"],
                        help="per-verb probing peak layers (for the vertical line)")
    parser.add_argument("--model", default="bert-base-uncased")
    args = parser.parse_args()

    peak_map = {}
    for item in args.peak_layers:
        v, l = item.split("=")
        peak_map[v] = int(l)

    tokenizer, model, device = load_bert(args.model)
    nlp = load_spacy()

    os.makedirs("output", exist_ok=True)
    all_disp = []
    for verb in args.verbs:
        print(f"\nCollecting all-layer vectors for '{verb}'...")
        vecs_by_level = collect_all_layers(args.csv, verb,
                                           tokenizer, model, device, nlp)
        if not vecs_by_level:
            print(f"[warn] no data for '{verb}'")
            continue

        disp_df = compute_dispersion(vecs_by_level)
        disp_df["verb"] = verb
        all_disp.append(disp_df)

        peak = peak_map.get(verb)
        plot_dispersion_lines(disp_df, verb, peak_layer=peak)
        plot_gap_curve(disp_df, verb, peak_layer=peak)

        # consistency check between the two metrics (strong negative = consistent)
        corr = disp_df[["kappa", "mean_dist"]].corr().iloc[0, 1]
        print(f"  corr(kappa, mean_dist) = {corr:.3f} "
              f"(strong negative => metrics agree => robust)")

    if all_disp:
        full = pd.concat(all_disp, ignore_index=True)
        full.to_csv("output/dispersion_all.csv", index=False)
        print("\nSaved: output/dispersion_all.csv")


if __name__ == "__main__":
    main()
