# -*- coding: utf-8 -*-
"""
phase2_probing.py
Layer-wise probing: for each of the 13 BERT layers, measure how well the verb
vector alone predicts the CEFR level, and identify the best-predicting layer.
That layer is then used as the basis for the Phase 3 vMF analysis.

Usage:
  python phase2_probing.py --csv JEFLL_ten_verbs.csv --verbs take make like
  (omit --verbs to probe all verbs together; add --per_verb to probe each verb)
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score

from shared_utils import (
    LEVEL_ORDER, load_data_with_ids, extract_verb_contexts,
    get_all_layer_vectors, load_bert, load_spacy,
)


# Target CEFR levels (after B1/B2 -> B1+): three-way classification.
TARGET_LEVELS = ["A1", "A2", "B1+"]


def build_dataset(csv_path, target_verbs, tokenizer, model, device, nlp,
                  max_per_cell=None, require_noun=False):
    """Build all-layer vectors X, CEFR labels y, and essay-id groups.

    max_per_cell caps the number of samples per verb x level (None = use all).
    Returns X (n, n_layers, 768), y (n,), groups (n,), and a meta DataFrame.
    """
    data = load_data_with_ids(csv_path)

    X_list, y_list, group_list, meta_rows = [], [], [], []
    for verb in target_verbs:
        if verb not in data:
            print(f"[warn] verb '{verb}' not found in CSV, skipped.")
            continue
        for level in TARGET_LEVELS:
            sents = data[verb].get(level, [])   # list of (sentence, file_id)
            if not sents:
                continue
            contexts = extract_verb_contexts(sents, verb, nlp, require_noun=require_noun)
            if max_per_cell is not None and len(contexts) > max_per_cell:
                idx = np.random.RandomState(42).choice(
                    len(contexts), size=max_per_cell, replace=False)
                contexts = [contexts[i] for i in idx]
            if not contexts:
                continue

            vecs = get_all_layer_vectors(contexts, tokenizer, model, device)
            if vecs.shape[0] == 0:
                continue

            # Align essay IDs to the vectors that were successfully extracted.
            file_ids = [c["file_id"] for c in contexts][:vecs.shape[0]]

            X_list.append(vecs)
            y_list.extend([level] * vecs.shape[0])
            group_list.extend(file_ids)
            meta_rows.extend([{"verb": verb, "level": level}] * vecs.shape[0])
            print(f"  {verb:6s} {level:4s}: {vecs.shape[0]:4d} vectors "
                  f"(layers={vecs.shape[1]})")

    X = np.concatenate(X_list, axis=0)
    y = np.array(y_list)
    groups = np.array(group_list)
    meta = pd.DataFrame(meta_rows)
    print(f"\nDataset built: X={X.shape}, y={y.shape}, groups={groups.shape}")
    print(f"  Unique essays (groups): {len(set(groups))}")
    print(f"  Class balance: {pd.Series(y).value_counts().to_dict()}")
    return X, y, groups, meta


def probe_each_layer(X, y, groups, n_splits=5):
    """Probe every layer with essay-level StratifiedGroupKFold CV.

    GroupKFold by essay ID prevents leakage: sentences from one essay must not
    split across train/test, or the classifier learns essay style, not verb
    meaning. A linear classifier is used on purpose, so the score reflects
    whether the layer encodes CEFR information in a linearly separable form.
    """
    n_layers = X.shape[1]
    rows = []
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for layer in range(n_layers):
        X_layer = X[:, layer, :]

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )

        y_pred = cross_val_predict(clf, X_layer, y, groups=groups, cv=sgkf)
        acc = accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        rows.append({"layer": layer, "accuracy": acc, "macro_f1": f1})
        print(f"  Layer {layer:2d}: acc={acc:.3f}  macro-F1={f1:.3f}")

    return pd.DataFrame(rows)


def plot_probing_curve(results_df, out_png="output/probing_curve.png",
                       chance_level=None):
    """Plot layer vs probing score and highlight the peak layer."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results_df["layer"], results_df["accuracy"],
            marker="o", linewidth=2.5, label="Accuracy")
    ax.plot(results_df["layer"], results_df["macro_f1"],
            marker="s", linewidth=2.5, label="Macro-F1")

    best = results_df.loc[results_df["accuracy"].idxmax()]
    ax.axvline(best["layer"], color="red", linestyle="--", alpha=0.6)
    ax.annotate(f"best layer = {int(best['layer'])}\nacc={best['accuracy']:.3f}",
                xy=(best["layer"], best["accuracy"]),
                xytext=(10, -30), textcoords="offset points",
                fontsize=11, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))

    if chance_level is not None:
        ax.axhline(chance_level, color="gray", linestyle=":",
                   label=f"chance ({chance_level:.2f})")

    ax.set_xlabel("BERT Layer (0 = embeddings, 12 = final)", fontsize=13)
    ax.set_ylabel("CEFR-level Classification Score", fontsize=13)
    ax.set_title("Layer-wise Probing: which layer encodes CEFR distinctions?",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(results_df["layer"])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}")
    return best


def probe_per_verb(X, y, groups, meta, n_splits=5):
    """Probe each layer separately per verb, so per-verb signals are not diluted
    by mixing verbs. Returns {verb: DataFrame(layer, accuracy, macro_f1)}."""
    results_by_verb = {}
    for verb in sorted(meta["verb"].unique()):
        mask = (meta["verb"] == verb).values
        Xv, yv, gv = X[mask], y[mask], groups[mask]
        n_cls = len(set(yv))
        chance = pd.Series(yv).value_counts(normalize=True).max()
        print(f"\n  -- verb '{verb}': n={len(yv)}, classes={sorted(set(yv))}, "
              f"chance={chance:.3f} --")
        if n_cls < 2:
            print(f"     [warn] only one class present, skipped")
            continue
        df_v = probe_each_layer(Xv, yv, gv, n_splits=n_splits)
        df_v["verb"] = verb
        df_v["chance"] = chance
        results_by_verb[verb] = df_v
    return results_by_verb


def plot_per_verb_curves(results_by_verb, out_png="output/probing_per_verb.png"):
    """Overlay per-verb layer-wise accuracy curves on one figure."""
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))

    markers = ["o", "s", "^", "D", "v", "p", "*"]
    for i, (verb, df_v) in enumerate(results_by_verb.items()):
        line, = ax.plot(df_v["layer"], df_v["accuracy"],
                        marker=markers[i % len(markers)], linewidth=2.5,
                        markersize=8, label=f"{verb} (acc)")
        ax.axhline(df_v["chance"].iloc[0], color=line.get_color(),
                   linestyle=":", alpha=0.5)
        best = df_v.loc[df_v["accuracy"].idxmax()]
        ax.scatter([best["layer"]], [best["accuracy"]], s=160,
                   facecolors="none", edgecolors=line.get_color(), linewidths=2)

    ax.set_xlabel("BERT Layer (0 = embeddings, 12 = final)", fontsize=13)
    ax.set_ylabel("CEFR-level Classification Accuracy", fontsize=13)
    ax.set_title("Layer-wise Probing by Verb\n"
                 "(dotted line = chance level for each verb; circle = peak layer)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(range(13))
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="JEFLL_ten_verbs.csv")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"],
                        help="verbs to probe (default: take make like)")
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--max_per_cell", type=int, default=200,
                        help="max samples per verb x level")
    parser.add_argument("--out", default="output/probing_curve.png")
    parser.add_argument("--per_verb", action="store_true",
                        help="probe each verb separately")
    args = parser.parse_args()

    tokenizer, model, device = load_bert(args.model)
    nlp = load_spacy()

    X, y, groups, meta = build_dataset(
        args.csv, args.verbs, tokenizer, model, device, nlp,
        max_per_cell=args.max_per_cell, require_noun=False,
    )

    if args.per_verb:
        print("\nProbing each layer, separately per verb (StratifiedGroupKFold)...")
        results_by_verb = probe_per_verb(X, y, groups, meta)

        plot_per_verb_curves(results_by_verb)

        all_df = pd.concat(results_by_verb.values(), ignore_index=True)
        all_df.to_csv("output/probing_per_verb.csv", index=False)
        print("Saved: output/probing_per_verb.csv")

        print("\nPeak layer per verb:")
        for verb, df_v in results_by_verb.items():
            best = df_v.loc[df_v["accuracy"].idxmax()]
            lift = best["accuracy"] - df_v["chance"].iloc[0]
            print(f"   {verb:6s}: best layer = {int(best['layer']):2d}, "
                  f"acc = {best['accuracy']:.3f} "
                  f"(chance {df_v['chance'].iloc[0]:.3f}, "
                  f"lift +{lift:.3f})")
        return

    # All-verbs mode. chance = share of the most frequent class.
    chance = pd.Series(y).value_counts(normalize=True).max()

    print("\nProbing each layer (essay-level StratifiedGroupKFold)...")
    results = probe_each_layer(X, y, groups)

    best = plot_probing_curve(results, out_png=args.out, chance_level=chance)
    results.to_csv("output/probing_results.csv", index=False)
    print(f"Saved: output/probing_results.csv")
    print(f"\nBest layer = {int(best['layer'])} "
          f"(accuracy={best['accuracy']:.3f}). Use this layer for Phase 3.")


if __name__ == "__main__":
    main()
