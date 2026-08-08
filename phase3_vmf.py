# -*- coding: utf-8 -*-
"""
phase3_vmf.py
vMF analysis around the best layer found in Phase 2. Produces:
  (A) a kappa heatmap over all layers x CEFR levels, and
  (B) a kappa line plot (verb x level) at the chosen layer, showing the
      expected A1 (high kappa, fixed) -> B1+ (low kappa, dispersed) pattern.

Usage:
  python phase3_vmf.py --csv JEFLL_ten_verbs.csv --verbs take make like --best_layer 9
  python phase3_vmf.py --compare --peak_layers take=6 make=11 like=11 --unified_layer 6
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
MIN_VECTORS = 2   # minimum samples for a meaningful vMF fit


def collect_layer_vectors(csv_path, target_verbs, tokenizer, model, device, nlp,
                          require_noun=False):
    """Return {verb: {level: ndarray(n, n_layers, 768)}}."""
    data = load_data_with_ids(csv_path)
    store = {}
    for verb in target_verbs:
        if verb not in data:
            print(f"[warn] verb '{verb}' not in CSV, skipped.")
            continue
        store[verb] = {}
        for level in TARGET_LEVELS:
            sents = data[verb].get(level, [])
            if not sents:
                continue
            contexts = extract_verb_contexts(sents, verb, nlp, require_noun=require_noun)
            if not contexts:
                continue
            vecs = get_all_layer_vectors(contexts, tokenizer, model, device)
            if vecs.shape[0] >= MIN_VECTORS:
                store[verb][level] = vecs
                print(f"  {verb:6s} {level:4s}: {vecs.shape[0]:4d} vectors "
                      f"x {vecs.shape[1]} layers")
    return store


def compute_kappa_table(store):
    """Compute kappa for all layer x verb x level. Returns long-form DataFrame
    with columns: verb, level, layer, kappa, n."""
    rows = []
    for verb, level_dict in store.items():
        for level, vecs in level_dict.items():
            n_layers = vecs.shape[1]
            for layer in range(n_layers):
                layer_vecs = vecs[:, layer, :]
                _, kappa = VonMisesFisherUtils.estimate_vmf_parameters(layer_vecs)
                rows.append({
                    "verb": verb, "level": level, "layer": layer,
                    "kappa": kappa if kappa is not None else np.nan,
                    "n": layer_vecs.shape[0],
                })
    return pd.DataFrame(rows)


def plot_kappa_heatmap(kappa_df, out_png="output/kappa_heatmap.png"):
    """kappa heatmap over all layers x levels (averaged across verbs)."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    pivot = (kappa_df.groupby(["layer", "level"])["kappa"]
             .mean().reset_index()
             .pivot(index="layer", columns="level", values="kappa"))
    pivot = pivot[[l for l in TARGET_LEVELS if l in pivot.columns]]

    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlabel("CEFR Level", fontsize=12)
    ax.set_ylabel("BERT Layer", fontsize=12)
    ax.set_title("Mean vMF kappa across layers and CEFR levels", fontsize=13,
                 fontweight="bold")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="kappa (concentration)")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def plot_best_layer_kappa(kappa_df, best_layer, out_png="output/kappa_best_layer.png"):
    """kappa line plot (verb x level) at the chosen best layer."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    sub = kappa_df[kappa_df["layer"] == best_layer].copy()

    fig, ax = plt.subplots(figsize=(9, 6))
    levels_present = [l for l in TARGET_LEVELS if l in sub["level"].unique()]
    markers = ["o", "s", "^", "D", "v", "p", "*", "h", "x", "+"]

    for i, verb in enumerate(sorted(sub["verb"].unique())):
        vsub = sub[sub["verb"] == verb].set_index("level").reindex(levels_present)
        ax.plot(levels_present, vsub["kappa"].values,
                marker=markers[i % len(markers)], linewidth=2.5,
                markersize=9, label=verb)
        for x, y in zip(levels_present, vsub["kappa"].values):
            if not np.isnan(y):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)

    ax.set_xlabel("CEFR Level", fontsize=13)
    ax.set_ylabel("vMF kappa (higher = more concentrated)", fontsize=13)
    ax.set_title(f"Semantic concentration kappa by CEFR level (Layer {best_layer})",
                 fontsize=14, fontweight="bold")
    ax.legend(title="Verb", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def plot_strategy_comparison(kappa_df, peak_layers, unified_layer,
                             out_png="output/kappa_strategy_comparison.png"):
    """Compare two layer-selection strategies side by side:
      left  = each verb's own peak layer,
      right = one unified layer for all verbs.
    Checks whether the A1->B1+ dispersion trend is robust to layer choice."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    levels = [l for l in TARGET_LEVELS if l in kappa_df["level"].unique()]
    verbs = sorted(kappa_df["verb"].unique())
    markers = ["o", "s", "^", "D", "v", "p", "*"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    # left: per-verb peak layer
    ax = axes[0]
    for i, verb in enumerate(verbs):
        layer = peak_layers.get(verb)
        if layer is None:
            continue
        sub = (kappa_df[(kappa_df["verb"] == verb) & (kappa_df["layer"] == layer)]
               .set_index("level").reindex(levels))
        ax.plot(levels, sub["kappa"].values, marker=markers[i % len(markers)],
                linewidth=2.5, markersize=9, label=f"{verb} (L{layer})")
        for x, y in zip(levels, sub["kappa"].values):
            if not np.isnan(y):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title("Strategy A: each verb's own peak layer",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CEFR Level", fontsize=12)
    ax.set_ylabel("vMF kappa (higher = more concentrated)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # right: unified layer
    ax = axes[1]
    for i, verb in enumerate(verbs):
        sub = (kappa_df[(kappa_df["verb"] == verb) & (kappa_df["layer"] == unified_layer)]
               .set_index("level").reindex(levels))
        ax.plot(levels, sub["kappa"].values, marker=markers[i % len(markers)],
                linewidth=2.5, markersize=9, label=f"{verb}")
        for x, y in zip(levels, sub["kappa"].values):
            if not np.isnan(y):
                ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8)
    ax.set_title(f"Strategy B: unified layer {unified_layer} for all verbs",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("CEFR Level", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle("vMF kappa by CEFR level: peak-layer vs unified-layer strategy",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")


def print_strategy_table(kappa_df, peak_layers, unified_layer):
    """Print kappa values for both strategies (down-arrow = dispersion)."""
    levels = [l for l in TARGET_LEVELS if l in kappa_df["level"].unique()]

    def kappa_at(verb, layer, level):
        row = kappa_df[(kappa_df["verb"] == verb) &
                       (kappa_df["layer"] == layer) &
                       (kappa_df["level"] == level)]
        return row["kappa"].iloc[0] if len(row) else float("nan")

    print("\n" + "=" * 64)
    print("kappa comparison (A1->B1+ decreasing = semantic dispersion)")
    print("=" * 64)
    for verb in sorted(kappa_df["verb"].unique()):
        pl = peak_layers.get(verb)
        print(f"\n[{verb}]")
        if pl is not None:
            vals = [kappa_at(verb, pl, lv) for lv in levels]
            trend = "disperse" if vals[0] > vals[-1] else "concentrate"
            print(f"  A) peak L{pl:<2d}: " +
                  "  ".join(f"{lv}={v:.0f}" for lv, v in zip(levels, vals)) +
                  f"   [{trend}]")
        vals = [kappa_at(verb, unified_layer, lv) for lv in levels]
        trend = "disperse" if vals[0] > vals[-1] else "concentrate"
        print(f"  B) unif L{unified_layer:<2d}: " +
              "  ".join(f"{lv}={v:.0f}" for lv, v in zip(levels, vals)) +
              f"   [{trend}]")
    print("\nIf both strategies agree on the trend, the conclusion is robust "
          "to layer choice.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="JEFLL_ten_verbs.csv")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--best_layer", type=int, default=None,
                        help="draw the line plot at a single best layer")
    parser.add_argument("--peak_layers", nargs="*", default=None,
                        help="per-verb peak layers, e.g. take=6 make=11 like=11")
    parser.add_argument("--unified_layer", type=int, default=None,
                        help="single layer used for all verbs, e.g. 6")
    parser.add_argument("--compare", action="store_true",
                        help="compare per-verb peak vs unified layer strategies")
    args = parser.parse_args()

    tokenizer, model, device = load_bert(args.model)
    nlp = load_spacy()

    print("\nCollecting all-layer vectors...")
    store = collect_layer_vectors(args.csv, args.verbs,
                                  tokenizer, model, device, nlp)

    print("\nComputing kappa for every layer x level...")
    kappa_df = compute_kappa_table(store)
    os.makedirs("output", exist_ok=True)
    kappa_df.to_csv("output/kappa_all_layers.csv", index=False)
    print("Saved: output/kappa_all_layers.csv")

    # (A) all-layer heatmap
    plot_kappa_heatmap(kappa_df)

    # (B) two-strategy comparison
    if args.compare:
        if args.peak_layers:
            peak_layers = {}
            for item in args.peak_layers:
                verb, layer = item.split("=")
                peak_layers[verb] = int(layer)
        else:
            peak_layers = {"take": 6, "make": 11, "like": 11}
            print(f"[info] --peak_layers not given; using defaults: {peak_layers}")

        unified = args.unified_layer if args.unified_layer is not None else 6
        print(f"\nComparing strategies: per-verb {peak_layers} vs unified {unified}")

        print_strategy_table(kappa_df, peak_layers, unified)
        plot_strategy_comparison(kappa_df, peak_layers, unified)
        return

    # (C) single best-layer line plot
    if args.best_layer is not None:
        plot_best_layer_kappa(kappa_df, args.best_layer)
    else:
        print("\n[info] neither --best_layer nor --compare given; line plot skipped.")
        print("   For the comparison: --compare --peak_layers take=6 make=11 like=11 --unified_layer 6")


if __name__ == "__main__":
    main()
