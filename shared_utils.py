# -*- coding: utf-8 -*-
"""
shared_utils.py
Shared helpers for the layer-wise verb-semantics pipeline (Phases 1-4).

Provides:
  - VonMisesFisherUtils : estimate the vMF concentration parameter kappa.
  - load_data / load_data_with_ids : read the CSV into {verb: {level: [...]}}.
  - extract_verb_contexts : locate the target verb token in each sentence.
  - get_all_layer_vectors : extract the verb-token vector from all BERT layers.

Note: B1 and B2 are merged into a single "B1+" level, following last year's setup.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import spacy


# Standard CEFR order. B1/B2 are collapsed into B1+.
LEVEL_ORDER = ["A1", "A2", "B1+", "C1", "C2"]


class VonMisesFisherUtils:
    """vMF utilities. Large kappa = concentrated (fixed meaning);
    small kappa = dispersed (diverse meaning)."""

    @staticmethod
    def estimate_vmf_parameters(vectors):
        """Estimate mean direction mu and concentration kappa from unit vectors.
        r (resultant length) near 1 => concentrated; near 0 => dispersed."""
        if len(vectors) == 0:
            return None, None
        mean_vector = np.mean(vectors, axis=0)
        r = np.linalg.norm(mean_vector)
        if r == 0:
            return None, 0
        mu = mean_vector / r
        d = vectors.shape[1]
        kappa = VonMisesFisherUtils._estimate_kappa(r, d)
        return mu, kappa

    @staticmethod
    def _estimate_kappa(r, d):
        """Banerjee et al. (2005) approximation: kappa ~= r(d - r^2) / (1 - r^2)."""
        if r >= 1.0:
            return float("inf")
        if r <= 0:
            return 0
        if d <= 2:
            return 2 * r / (1 - r ** 2)
        return (r * (d - r ** 2)) / (1 - r ** 2)


def load_data(csv_path, verbose=True):
    """Read the CSV into {verb: {level: [sentences]}}.
    Requires at least the columns: verb, cefr, sentence."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    verbs = df["verb"].str.lower().str.strip().unique()
    if verbose:
        print(f"Found {len(verbs)} unique verbs: {list(verbs)}")

    data = {}
    for verb in verbs:
        verb_df = df[df["verb"].str.lower().str.strip() == verb]
        data[verb] = {}
        for level in verb_df["cefr"].str.upper().str.strip().unique():
            if level in ["B1", "B2"]:
                mapped = "B1+"
            elif level in ["A1", "A2", "C1", "C2"]:
                mapped = level
            else:
                continue
            sents = (
                verb_df[verb_df["cefr"].str.upper().str.strip() == level]["sentence"]
                .dropna().str.strip().tolist()
            )
            sents = [s for s in sents if s]
            data[verb].setdefault(mapped, []).extend(sents)

    if verbose:
        print("\nData summary (after B1/B2 -> B1+):")
        for verb in data:
            print(f"  {verb}:")
            for level in LEVEL_ORDER:
                if level in data[verb]:
                    print(f"    {level}: {len(data[verb][level])} sentences")
    return data


def load_data_with_ids(csv_path, verbose=True):
    """Same as load_data, but each item is a (sentence, file_id) tuple so the
    essay ID is retained for essay-level GroupKFold in Phase 2.
    Requires the columns: file, cefr, sentence, verb."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()

    verbs = df["verb"].str.lower().str.strip().unique()
    if verbose:
        print(f"Found {len(verbs)} unique verbs: {list(verbs)}")

    data = {}
    for verb in verbs:
        verb_df = df[df["verb"].str.lower().str.strip() == verb]
        data[verb] = {}
        for level in verb_df["cefr"].str.upper().str.strip().unique():
            if level in ["B1", "B2"]:
                mapped = "B1+"
            elif level in ["A1", "A2", "C1", "C2"]:
                mapped = level
            else:
                continue
            cell = verb_df[verb_df["cefr"].str.upper().str.strip() == level]
            pairs = [
                (str(s).strip(), str(f))
                for s, f in zip(cell["sentence"], cell["file"])
                if pd.notna(s) and str(s).strip()
            ]
            data[verb].setdefault(mapped, []).extend(pairs)

    if verbose:
        print("\nData summary (with file IDs, B1/B2 -> B1+):")
        for verb in data:
            print(f"  {verb}:")
            for level in LEVEL_ORDER:
                if level in data[verb]:
                    print(f"    {level}: {len(data[verb][level])} sentences")
    return data


def extract_verb_contexts(sentences, verb, nlp, require_noun=False):
    """Keep sentences containing the target verb (lemma match & POS=VERB).
    require_noun=True keeps only sentences with a noun object (dobj/attr/pobj).
    Accepts either plain sentences or (sentence, file_id) tuples."""
    contexts = []
    for item in sentences:
        if isinstance(item, tuple):
            sentence, file_id = item
        else:
            sentence, file_id = item, None

        doc = nlp(sentence)
        for token in doc:
            if token.lemma_.lower() == verb.lower() and token.pos_ == "VERB":
                noun_text = None
                for child in token.children:
                    if child.dep_ in {"dobj", "attr", "pobj"} and child.pos_ in {"NOUN", "PROPN"}:
                        noun_text = child.text
                        break
                if require_noun and noun_text is None:
                    break
                contexts.append({
                    "sentence": sentence,
                    "verb_idx": token.idx,       # character offset (start)
                    "verb_text": token.text,
                    "noun_text": noun_text,
                    "file_id": file_id,
                })
                break  # first occurrence per sentence only
    return contexts


def get_all_layer_vectors(contexts, tokenizer, model, device):
    """Extract the target-verb vector from every BERT layer for each context.
    Returns an array of shape (n_contexts, n_layers, hidden_dim);
    for BERT-base that is (n, 13, 768). Sub-word pieces of the verb are averaged,
    and each vector is L2-normalised to a unit vector."""
    all_vectors = []

    for ctx in contexts:
        text = ctx["sentence"]
        v_start = ctx["verb_idx"]
        v_end = ctx["verb_idx"] + len(ctx["verb_text"])

        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()
        model_inputs = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**model_inputs)
            hidden_states = outputs.hidden_states  # tuple of (1, seq_len, 768)

        # tokens whose character span overlaps the verb span
        verb_token_indices = [
            i for i, (s, e) in enumerate(offsets)
            if not (e <= v_start or s >= v_end) and not (s == 0 and e == 0)
        ]
        if not verb_token_indices:
            continue

        per_layer = []
        for layer_tensor in hidden_states:
            vec = layer_tensor[0][verb_token_indices].mean(dim=0)  # (768,)
            vec = vec.detach().cpu().numpy()
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            per_layer.append(vec)

        all_vectors.append(np.stack(per_layer, axis=0))  # (n_layers, 768)

    if not all_vectors:
        return np.empty((0, 0, 0))
    return np.stack(all_vectors, axis=0)


def load_bert(model_name="bert-base-uncased"):
    """Load a BERT model and tokenizer with output_hidden_states=True."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return tokenizer, model, device


def load_spacy():
    """Load the spaCy English model (POS tagging and dependency parsing)."""
    return spacy.load("en_core_web_sm")
