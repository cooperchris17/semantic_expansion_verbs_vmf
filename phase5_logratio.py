# -*- coding: utf-8 -*-
"""
phase5_logratio.py
Anisotropy-aware comparison (Method 4.5).

Raw kappa is inflated by anisotropy, so between-level concentration is compared
with the scale-free log-ratio ln(kappa_from / kappa_to). A shared multiplicative
component (the anisotropy factor) cancels in the ratio, so the log-ratio reflects
the genuine change in concentration rather than the ambient scale.

Sign convention: pairs run in ascending CEFR order (from -> to), and the ratio is
ln(kappa_from / kappa_to). A POSITIVE value means concentration drops as the level
rises (kappa_from > kappa_to), i.e. the higher level is more dispersed -- the
predicted direction.

Confidence: an essay-level cluster bootstrap (B=2000, matching Method 4.4) resamples
the two levels' essays independently and recomputes the log-ratio each time. A pair
is treated as reliable when its 95% CI excludes 0.

Usage:
  python phase5_logratio.py --csv JEFLL_ten_verbs.csv --verbs take make like \
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
# level pairs compared (from -> to), ascending CEFR order
LEVEL_PAIRS = [("A1", "A2"), ("A2", "B1+"), ("A1", "B1+")]
MIN_VECTORS = 5


def collect_vectors_with_ids(csv_path, target_verbs, layer_map,
                             tokenizer, model, device, nlp, require_noun=False):
    """Per verb x level, collect the anchor-layer vectors and their essay IDs.
    Returns {verb: {level: (vecs(n,768), file_ids(n,))}}. Same shape as phase3b."""
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
            vecs = all_vecs[:, layer, :]
            file_ids = np.array([c["file_id"] for c in contexts][:vecs.shape[0]])
            store[verb][level] = (vecs, file_ids)
            print(f"  {verb:6s} {level:4s} (L{layer}): {vecs.shape[0]:4d} vectors")
    return store


def _kappa(vecs):
    """kappa point estimate for a set of unit vectors (None if undefined)."""
    _, k = VonMisesFisherUtils.estimate_vmf_parameters(vecs)
    return k


def _resample_kappa(vecs, file_ids, rng, cluster):
    """One bootstrap kappa: essay-level (cluster) or sentence-level resample."""
    if cluster:
        unique_files = np.unique(file_ids)
        file_to_rows = {f: np.where(file_ids == f)[0] for f in unique_files}
        sampled = rng.choice(unique_files, size=len(unique_files), replace=True)
        rows = np.concatenate([file_to_rows[f] for f in sampled])
        return _kappa(vecs[rows])
    idx = rng.randint(0, vecs.shape[0], size=vecs.shape[0])
    return _kappa(vecs[idx])


def logratio_ci(vecs_from, ids_from, vecs_to, ids_to,
                n_boot=2000, cluster=True, ci=95, seed=42):
    """Log-ratio ln(kappa_from / kappa_to) with a two-sample bootstrap CI.
    The two levels are resampled independently each iteration."""
    k_from = _kappa(vecs_from)
    k_to = _kappa(vecs_to)
    if not (k_from and k_to and np.isfinite(k_from) and np.isfinite(k_to)
            and k_from > 0 and k_to > 0):
        return np.nan, np.nan, np.nan, k_from, k_to

    point = np.log(k_from / k_to)
    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_boot):
        kf = _resample_kappa(vecs_from, ids_from, rng, cluster)
        kt = _resample_kappa(vecs_to, ids_to, rng, cluster)
        if kf and kt and np.isfinite(kf) and np.isfinite(kt) and kf > 0 and kt > 0:
            boot.append(np.log(kf / kt))
    boot = np.array(boot)
    lo = np.percentile(boot, (100 - ci) / 2)
    hi = np.percentile(boot, 100 - (100 - ci) / 2)
    return point, lo, hi, k_from, k_to


def run_all(store, n_boot=2000, cluster=True):
    """Compute the log-ratio and its CI for every verb and level pair."""
    rows = []
    for verb, level_dict in store.items():
        for lv_from, lv_to in LEVEL_PAIRS:
            if lv_from not in level_dict or lv_to not in level_dict:
                continue
            vf, idf = level_dict[lv_from]
            vt, idt = level_dict[lv_to]
            point, lo, hi, kf, kt = logratio_ci(
                vf, idf, vt, idt, n_boot=n_boot, cluster=cluster)
            rows.append({
                "verb": verb, "from": lv_from, "to": lv_to,
                "kappa_from": kf, "kappa_to": kt,
                "log_ratio": point, "ci_low": lo, "ci_high": hi,
                "n_from": vf.shape[0], "n_to": vt.shape[0],
            })
            excludes0 = not (lo <= 0 <= hi) if np.isfinite(lo) else False
            tag = "significant" if excludes0 else "n.s."
            print(f"  {verb:6s} {lv_from:>3s}->{lv_to:<3s}: "
                  f"ln(k_from/k_to)={point:+.3f}  "
                  f"95%CI=[{lo:+.3f}, {hi:+.3f}]  "
                  f"(k {kf:.0f} vs {kt:.0f})  -> {tag}")
    return pd.DataFrame(rows)


def plot_logratio(df, out_png="output/logratio_ci.png", cluster=True):
    """Log-ratio per level pair, grouped by verb, with bootstrap error bars.
    Values above the zero line indicate dispersion as the CEFR level rises."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    verbs = sorted(df["verb"].unique())
    pair_labels = [f"{a}->{b}" for a, b in LEVEL_PAIRS]
    x = np.arange(len(pair_labels))
    width = 0.8 / max(len(verbs), 1)
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, verb in enumerate(verbs):
        sub = df[df["verb"] == verb].set_index(
            df[df["verb"] == verb].apply(lambda r: f"{r['from']}->{r['to']}", axis=1)
        ).reindex(pair_labels)
        y = sub["log_ratio"].values
        lo = sub["ci_low"].values
        hi = sub["ci_high"].values
        yerr = np.vstack([y - lo, hi - y])
        ax.errorbar(x + (i - (len(verbs) - 1) / 2) * width, y, yerr=yerr,
                    marker=markers[i % len(markers)], markersize=9,
                    linewidth=0, elinewidth=2, capsize=6, capthick=2,
                    label=verb)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=12)
    method = "essay-level (cluster)" if cluster else "sentence-level"
    ax.set_xlabel("CEFR level pair (from -> to)", fontsize=13)
    ax.set_ylabel("ln(kappa_from / kappa_to)\n(> 0 = dispersion as level rises)",
                  fontsize=12)
    ax.set_title(f"Anisotropy-aware concentration change: log-ratio with 95% CI "
                 f"({method})", fontsize=13, fontweight="bold")
    ax.legend(title="Verb", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="JEFLL_ten_verbs.csv")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--layers", nargs="*", default=["take=6", "make=11", "like=11"],
                        help="per-verb anchor layer, e.g. take=6 make=11 like=11")
    parser.add_argument("--n_boot", type=int, default=2000,
                        help="number of bootstrap iterations (default 2000)")
    parser.add_argument("--cluster", action="store_true",
                        help="essay-level bootstrap (matches Method 4.4); else sentence-level")
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

    print("\nCollecting vectors at anchor layers...")
    store = collect_vectors_with_ids(args.csv, args.verbs, layer_map,
                                     tokenizer, model, device, nlp)

    print(f"\nComputing log-ratios with bootstrap CI (B={args.n_boot})...")
    df = run_all(store, n_boot=args.n_boot, cluster=args.cluster)

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/logratio_ci.csv", index=False)
    print("Saved: output/logratio_ci.csv")

    plot_logratio(df, cluster=args.cluster)


if __name__ == "__main__":
    main()
