# -*- coding: utf-8 -*-
"""
phase3b_bootstrap.py
Bootstrap 95% confidence intervals for vMF kappa. Non-overlapping CIs between
adjacent CEFR levels support the claim that the A1->B1+ decrease in kappa is
real rather than sampling noise.

Two bootstrap modes:
  - sentence-level : resample vectors one by one.
  - essay-level (cluster) : resample essays, take all sentences of each; more
    conservative, since sentences within an essay are not independent.

Usage:
  python phase3b_bootstrap.py --csv JEFLL_ten_verbs.csv --verbs take make like \
      --layers take=6 make=11 like=11 --n_boot 2000 --cluster
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
MIN_VECTORS = 5


def collect_vectors_with_ids(csv_path, target_verbs, layer_map,
                             tokenizer, model, device, nlp, require_noun=False):
    """Collect, per verb x level, the vectors at the specified layer plus the
    essay IDs. Returns {verb: {level: (vecs(n,768), file_ids(n,))}}."""
    data = load_data_with_ids(csv_path)
    store = {}
    for verb in target_verbs:
        if verb not in data:
            print(f"[warn] verb '{verb}' not in CSV, skipped.")
            continue
        layer = layer_map[verb]
        store[verb] = {}
        for level in TARGET_LEVELS:
            sents = data[verb].get(level, [])
            if not sents:
                continue
            contexts = extract_verb_contexts(sents, verb, nlp, require_noun=require_noun)
            if not contexts:
                continue
            all_vecs = get_all_layer_vectors(contexts, tokenizer, model, device)
            if all_vecs.shape[0] < MIN_VECTORS:
                continue
            vecs = all_vecs[:, layer, :]                       # (n, 768)
            file_ids = np.array([c["file_id"] for c in contexts][:vecs.shape[0]])
            store[verb][level] = (vecs, file_ids)
            print(f"  {verb:6s} {level:4s} (L{layer}): {vecs.shape[0]:4d} vectors")
    return store


def bootstrap_kappa_ci(vecs, file_ids, n_boot=2000, cluster=True,
                       ci=95, seed=42):
    """Bootstrap CI for kappa.
    cluster=True resamples essays; cluster=False resamples sentences.
    Returns (kappa_point, ci_low, ci_high, boot_samples)."""
    rng = np.random.RandomState(seed)
    n = vecs.shape[0]

    _, kappa_point = VonMisesFisherUtils.estimate_vmf_parameters(vecs)

    boot_kappas = []
    if cluster:
        unique_files = np.unique(file_ids)
        file_to_rows = {f: np.where(file_ids == f)[0] for f in unique_files}
        n_clusters = len(unique_files)

        for _ in range(n_boot):
            sampled_files = rng.choice(unique_files, size=n_clusters, replace=True)
            rows = np.concatenate([file_to_rows[f] for f in sampled_files])
            sample = vecs[rows]
            _, k = VonMisesFisherUtils.estimate_vmf_parameters(sample)
            if k is not None and np.isfinite(k):
                boot_kappas.append(k)
    else:
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            sample = vecs[idx]
            _, k = VonMisesFisherUtils.estimate_vmf_parameters(sample)
            if k is not None and np.isfinite(k):
                boot_kappas.append(k)

    boot_kappas = np.array(boot_kappas)
    lo = np.percentile(boot_kappas, (100 - ci) / 2)
    hi = np.percentile(boot_kappas, 100 - (100 - ci) / 2)
    return kappa_point, lo, hi, boot_kappas


def run_all(store, n_boot=2000, cluster=True):
    """Compute CIs for every verb x level and collect into a DataFrame."""
    rows = []
    for verb, level_dict in store.items():
        for level in TARGET_LEVELS:
            if level not in level_dict:
                continue
            vecs, file_ids = level_dict[level]
            point, lo, hi, _ = bootstrap_kappa_ci(
                vecs, file_ids, n_boot=n_boot, cluster=cluster)
            rows.append({
                "verb": verb, "level": level,
                "kappa": point, "ci_low": lo, "ci_high": hi,
                "n": vecs.shape[0],
                "n_essays": len(np.unique(file_ids)),
            })
            print(f"  {verb:6s} {level:4s}: kappa={point:7.1f}  "
                  f"95%CI=[{lo:7.1f}, {hi:7.1f}]  "
                  f"n={vecs.shape[0]} ({len(np.unique(file_ids))} essays)")
    return pd.DataFrame(rows)


def plot_with_ci(ci_df, out_png="output/kappa_with_ci.png", cluster=True):
    """Plot kappa with error bars from the bootstrap CIs."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    levels = [l for l in TARGET_LEVELS if l in ci_df["level"].unique()]
    verbs = sorted(ci_df["verb"].unique())
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, verb in enumerate(verbs):
        sub = ci_df[ci_df["verb"] == verb].set_index("level").reindex(levels)
        y = sub["kappa"].values
        lo = sub["ci_low"].values
        hi = sub["ci_high"].values
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(levels, y, yerr=yerr, marker=markers[i % len(markers)],
                    markersize=9, linewidth=2.5, capsize=6, capthick=2,
                    label=verb)
        for x, yi in zip(levels, y):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.0f}", (x, yi), textcoords="offset points",
                            xytext=(8, 0), fontsize=8)

    method = "essay-level (cluster)" if cluster else "sentence-level"
    ax.set_xlabel("CEFR Level", fontsize=13)
    ax.set_ylabel("vMF kappa (higher = more concentrated)", fontsize=13)
    ax.set_title(f"vMF kappa with 95% bootstrap CI ({method}, B={ci_df.attrs.get('n_boot','?')})",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


def print_significance(ci_df):
    """Report whether adjacent-level CIs overlap (non-overlap => significant)."""
    print("\n" + "=" * 64)
    print("Adjacent-level differences (non-overlapping CI = significant)")
    print("=" * 64)
    levels = [l for l in TARGET_LEVELS if l in ci_df["level"].unique()]
    for verb in sorted(ci_df["verb"].unique()):
        sub = ci_df[ci_df["verb"] == verb].set_index("level")
        print(f"\n[{verb}]")
        for a, b in zip(levels[:-1], levels[1:]):
            if a not in sub.index or b not in sub.index:
                continue
            a_lo, a_hi = sub.loc[a, "ci_low"], sub.loc[a, "ci_high"]
            b_lo, b_hi = sub.loc[b, "ci_low"], sub.loc[b, "ci_high"]
            overlap = not (a_lo > b_hi or b_lo > a_hi)
            ka, kb = sub.loc[a, "kappa"], sub.loc[b, "kappa"]
            direction = "down" if ka > kb else "up"
            verdict = "overlap (n.s.)" if overlap else "no overlap (significant)"
            print(f"  {a}->{b}: kappa {ka:.0f} {direction} {kb:.0f}   "
                  f"CI[{a_lo:.0f},{a_hi:.0f}] vs [{b_lo:.0f},{b_hi:.0f}]  -> {verdict}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="JEFLL_ten_verbs.csv")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--layers", nargs="*", default=["take=6", "make=11", "like=11"],
                        help="per-verb layer for the CI, e.g. take=6 make=11 like=11")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="number of bootstrap iterations (default 2000)")
    parser.add_argument("--cluster", action="store_true",
                        help="essay-level bootstrap (recommended); else sentence-level")
    args = parser.parse_args()

    layer_map = {}
    for item in args.layers:
        v, l = item.split("=")
        layer_map[v] = int(l)
    print(f"Layer map: {layer_map}")
    print(f"Bootstrap: {'essay-level (cluster)' if args.cluster else 'sentence-level'}, "
          f"B={args.n_boot}")

    tokenizer, model, device = load_bert(args.model)
    nlp = load_spacy()

    print("\nCollecting vectors at specified layers...")
    store = collect_vectors_with_ids(args.csv, args.verbs, layer_map,
                                     tokenizer, model, device, nlp)

    print(f"\nBootstrapping kappa (B={args.n_boot})...")
    ci_df = run_all(store, n_boot=args.n_boot, cluster=args.cluster)
    ci_df.attrs["n_boot"] = args.n_boot

    os.makedirs("output", exist_ok=True)
    ci_df.to_csv("output/kappa_with_ci.csv", index=False)
    print("Saved: output/kappa_with_ci.csv")

    plot_with_ci(ci_df, cluster=args.cluster)
    print_significance(ci_df)


if __name__ == "__main__":
    main()
