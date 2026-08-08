# -*- coding: utf-8 -*-
"""
This script classifies the complementation patterns of the target verbs (take / make / like) 
and exports frequency tables to Excel.

    noun / proper_noun    nominal object (dobj, attr, pobj, dative)
    coordinated_noun      noun coordinated with the above ("rice and bread")
    pronoun               pronominal object ("like it")
    gerund                gerundial complement (xcomp, VBG; "like swimming")
    infinitive            infinitival complement (xcomp, VB; "like to swim")
    clausal               clausal complement (ccomp; "like that we can ...")
    predicate_adjective   adjectival complement (acomp; "make sure")
    passive_subject       subject of a passive ("bread was made")
    phrasal_only          no nominal or clausal argument, particle only ("take off")
    none                  none of the above

Each verb token gets a single primary category, chosen by CONSTRUCTION_PRIORITY
(most informative first). Coordinated nouns other than the primary one are
counted only in the noun collocate table, so category counts are not inflated.
Only the first occurrence of the target verb in a sentence is analysed.

Outputs (all under output/):
  constructions_all.csv        one row per verb occurrence
  category_summary.csv         category counts and percentages by verb x level
  relativization_summary.csv   relative-clause object recoveries by verb x level
  extract_constructions.xlsx   the above as sheets, plus:
      raw_all                  full raw data
      noun_collocates          noun + proper_noun + coordinated_noun frequencies
      verbal_complements       gerund and infinitive complement frequencies

Usage:
  python extract_constructions.py --csv JEFLL_ten_verbs.csv --verbs take make like
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from shared_utils import load_data_with_ids, load_spacy

TARGET_LEVELS = ["A1", "A2", "B1+"]

# Order in which the primary category is chosen (earlier = higher priority).
# phrasal_only / none are not listed: they are the fallback when no candidate
# is found at all.
CONSTRUCTION_PRIORITY = [
    "noun", "proper_noun", "pronoun", "coordinated_noun",
    "gerund", "infinitive", "clausal",
    "predicate_adjective", "passive_subject",
]

NOUN_CATEGORIES = {"noun", "proper_noun", "coordinated_noun"}
VERBAL_CATEGORIES = {"gerund", "infinitive"}


def extract_constructions(sentences, verb, nlp):
    """
    Classify each occurrence of `verb` in `sentences` (a list of strings, or of
    (sentence, file_id) tuples) and return one dict per occurrence.
    """
    contexts = []
    for item in sentences:
        if isinstance(item, tuple):
            sentence, file_id = item
        else:
            sentence, file_id = item, None

        doc = nlp(sentence)
        for token in doc:
            if token.lemma_.lower() != verb.lower() or token.pos_ != "VERB":
                continue

            particle = None
            candidates = []  # [(category, text_or_None), ...]
            relativized = False

            # Relative-clause object gap.
            # If the target verb heads a relative clause (dep_ == "relcl") that
            # already contains a subject, the missing argument is most likely the
            # object, and token.head is the antecedent, i.e. the real semantic
            # object. Recover the antecedent as noun/proper_noun rather than
            # counting "that"/"which" as a pronoun.
            #   "the bread that I make"   -> relcl, head "bread": object gap.
            #   "the man who makes bread" -> relcl, but "who" is the subject, so
            #                                this is a subject gap and is skipped;
            #                                "bread" is caught as an ordinary dobj.
            if token.dep_ == "relcl" and token.head.pos_ in {"NOUN", "PROPN"}:
                # A wh-word subject (who/that, WP/WDT) marks a subject relative
                # clause, so there is no object gap to recover.
                subject_is_wh = any(
                    c.dep_ in {"nsubj", "nsubjpass"} and c.tag_ in {"WP", "WDT"}
                    for c in token.children
                )
                # An ordinary, non-wh subject, e.g. "I" in "the bread that I make".
                has_real_subject = any(
                    c.dep_ in {"nsubj", "nsubjpass"} and c.tag_ not in {"WP", "WDT"}
                    for c in token.children
                )
                # If the object slot is already filled by a real noun, the gap is
                # not an object gap (e.g. an adverbial relative clause).
                object_slot_filled = any(
                    c.dep_ in {"dobj", "attr", "pobj", "dative"}
                    and c.pos_ in {"NOUN", "PROPN"}
                    for c in token.children
                )
                if not subject_is_wh and has_real_subject and not object_slot_filled:
                    antecedent = token.head
                    cat = "proper_noun" if antecedent.pos_ == "PROPN" else "noun"
                    candidates.append((cat, antecedent.text.lower()))
                    relativized = True

            for child in token.children:
                if child.dep_ == "prt":
                    particle = child.text.lower()

                elif child.dep_ in {"dobj", "attr", "pobj", "dative"}:
                    if child.pos_ == "NOUN":
                        candidates.append(("noun", child.text.lower()))
                        for conj in child.conjuncts:
                            if conj.pos_ in {"NOUN", "PROPN"}:
                                candidates.append(("coordinated_noun", conj.text.lower()))
                    elif child.pos_ == "PROPN":
                        candidates.append(("proper_noun", child.text.lower()))
                        for conj in child.conjuncts:
                            if conj.pos_ in {"NOUN", "PROPN"}:
                                candidates.append(("coordinated_noun", conj.text.lower()))
                    elif child.pos_ == "PRON":
                        # If relativized, this PRON is most likely the relative
                        # pronoun whose antecedent was recovered above, so it is
                        # not counted again as a pronoun object.
                        if not relativized:
                            candidates.append(("pronoun", child.text.lower()))

                elif child.dep_ == "xcomp":
                    if child.tag_ == "VBG":
                        candidates.append(("gerund", child.lemma_.lower()))
                    elif child.tag_ == "VB":
                        candidates.append(("infinitive", child.lemma_.lower()))

                elif child.dep_ == "ccomp":
                    candidates.append(("clausal", None))

                elif child.dep_ == "acomp":
                    candidates.append(("predicate_adjective", child.text.lower()))

                elif child.dep_ == "nsubjpass":
                    candidates.append(("passive_subject", child.text.lower()))

            # Primary category = first match in priority order.
            primary = None
            for cat in CONSTRUCTION_PRIORITY:
                match = next((c for c in candidates if c[0] == cat), None)
                if match:
                    primary = match
                    break

            if primary is None:
                primary = ("phrasal_only", None) if particle else ("none", None)

            category, collocate = primary

            # Coordinated nouns other than the primary one: noun collocate table only.
            extra_nouns = [c[1] for c in candidates
                           if c[0] == "coordinated_noun" and c != primary]

            contexts.append({
                "verb": verb,
                "sentence": sentence,
                "file_id": file_id,
                "verb_text": token.text,
                "particle": particle,
                "category": category,
                "collocate": collocate,
                "extra_nouns": "; ".join(extra_nouns) if extra_nouns else None,
                "relativized": relativized,
            })
            break  # only the first occurrence per sentence

    return contexts


def build_all_contexts(csv_path, verbs, nlp):
    """Run extract_constructions over all verbs and levels; return a long DataFrame."""
    data = load_data_with_ids(csv_path)
    rows = []
    for verb in verbs:
        if verb not in data:
            print(f"⚠️ verb '{verb}' not found in CSV, skipped.")
            continue
        for level in TARGET_LEVELS:
            sents = data[verb].get(level, [])
            if not sents:
                continue
            contexts = extract_constructions(sents, verb, nlp)
            for c in contexts:
                c["level"] = level
                rows.append(c)
            print(f"  {verb:6s} {level:4s}: {len(contexts):4d} occurrences processed")
    return pd.DataFrame(rows)


def load_topics(csv_path):
    """
    Return a (file_id, sentence) -> topic lookup table, or None if the source CSV
    has no 'topic' column (attach_topics then skips the join).

    file_id is cast to str to match load_data_with_ids; an int64/object mismatch
    would otherwise break the merge.

    The source CSV has one row per (verb, sentence), so a sentence containing
    more than one target verb ("I take money and make food") appears on several
    rows with the same (file_id, sentence). Those duplicates are collapsed here
    to keep the merge one-to-one and avoid fan-out.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    if "topic" not in df.columns:
        print("ℹ️ No 'topic' column found in the CSV; topic metadata will be skipped.")
        return None
    topics = df[["file", "sentence", "topic"]].copy()
    topics.columns = ["file_id", "sentence", "topic"]
    topics["file_id"] = topics["file_id"].astype(str)
    topics["sentence"] = topics["sentence"].astype(str).str.strip()

    # Fully identical rows (same file_id/sentence/topic) collapse cleanly.
    topics = topics.drop_duplicates(subset=["file_id", "sentence", "topic"])

    # If a (file_id, sentence) still maps to conflicting topic values, warn and
    # keep the first so the merge stays one-to-one.
    dup_mask = topics.duplicated(subset=["file_id", "sentence"], keep=False)
    if dup_mask.any():
        n_conflicting = topics.loc[dup_mask, ["file_id", "sentence"]].drop_duplicates().shape[0]
        print(f"⚠️ {n_conflicting} (file_id, sentence) pairs have conflicting topic "
              f"values; keeping the first.")
        topics = topics.drop_duplicates(subset=["file_id", "sentence"], keep="first")

    return topics


def attach_topics(raw_df, csv_path):
    """
    Add a topic column to raw_df, joining on (file_id, sentence) with file_id
    cast to str on both sides. Unmatched rows and any change in row count are
    reported so unexpected losses or fan-out are visible.
    """
    topics = load_topics(csv_path)
    if topics is None:
        raw_df["topic"] = None
        return raw_df

    raw_df = raw_df.copy()
    raw_df["file_id"] = raw_df["file_id"].astype(str)
    raw_df["sentence"] = raw_df["sentence"].astype(str).str.strip()

    n_before = len(raw_df)
    merged = raw_df.merge(topics, on=["file_id", "sentence"], how="left")
    n_after = len(merged)
    if n_after != n_before:
        print(f"⚠️ Row count changed during merge: {n_before} -> {n_after}. "
              f"The topic join key may not be unique.")

    n_missing = merged["topic"].isna().sum()
    if n_missing:
        print(f"⚠️ {n_missing} / {len(merged)} rows could not be matched to a topic "
              f"(likely a file_id/sentence mismatch).")
    return merged


def build_category_summary(raw_df):
    """Counts and percentages by verb x level x category."""
    counts = (raw_df.groupby(["verb", "level", "category"])
              .size().reset_index(name="n"))
    totals = (raw_df.groupby(["verb", "level"])
              .size().reset_index(name="total"))
    summary = counts.merge(totals, on=["verb", "level"])
    summary["pct"] = (summary["n"] / summary["total"] * 100).round(1)
    summary = summary.sort_values(["verb", "level", "n"], ascending=[True, True, False])
    return summary[["verb", "level", "category", "n", "total", "pct"]]


def build_noun_collocates(raw_df):
    """
    Noun collocate frequencies (noun + proper_noun + primary coordinated_noun +
    extra_nouns).

    Two percentages are reported:
      pct_of_sentences     share of all occurrences for that verb x level; same
                           denominator as category_summary, so the two sheets are
                           directly comparable
      pct_within_category  share of all noun collocates at that level, i.e. the
                           relative frequency used for ranking
    """
    rows = []
    for _, r in raw_df.iterrows():
        if r["category"] in NOUN_CATEGORIES and r["collocate"]:
            rows.append({"verb": r["verb"], "level": r["level"], "noun": r["collocate"]})
        if pd.notna(r["extra_nouns"]):
            for noun in r["extra_nouns"].split("; "):
                rows.append({"verb": r["verb"], "level": r["level"], "noun": noun})
    if not rows:
        return pd.DataFrame(columns=[
            "verb", "level", "noun", "n", "pct_of_sentences", "pct_within_category",
        ])
    df = pd.DataFrame(rows)
    freq = (df.groupby(["verb", "level", "noun"]).size()
            .reset_index(name="n"))

    total_sents = (raw_df.groupby(["verb", "level"]).size()
                   .reset_index(name="total_sentences"))
    freq = freq.merge(total_sents, on=["verb", "level"])
    freq["pct_of_sentences"] = (freq["n"] / freq["total_sentences"] * 100).round(1)

    category_totals = freq.groupby(["verb", "level"])["n"].transform("sum")
    freq["pct_within_category"] = (freq["n"] / category_totals * 100).round(1)

    freq = freq.sort_values(["verb", "level", "n"], ascending=[True, True, False])
    return freq[["verb", "level", "noun", "n",
                 "pct_of_sentences", "pct_within_category", "total_sentences"]]


def build_verbal_complements(raw_df):
    """
    Gerund and infinitive complement frequencies ("like playing", "like to play").

    pct_of_sentences     share of all occurrences for that verb x level
    pct_within_category  share within the same complement type; gerund and
                         infinitive are normalised against separate totals
    """
    sub = raw_df[raw_df["category"].isin(VERBAL_CATEGORIES) & raw_df["collocate"].notna()]
    if sub.empty:
        return pd.DataFrame(columns=[
            "verb", "level", "category", "complement", "n",
            "pct_of_sentences", "pct_within_category",
        ])
    freq = (sub.rename(columns={"collocate": "complement"})
            .groupby(["verb", "level", "category", "complement"]).size()
            .reset_index(name="n"))

    total_sents = (raw_df.groupby(["verb", "level"]).size()
                   .reset_index(name="total_sentences"))
    freq = freq.merge(total_sents, on=["verb", "level"])
    freq["pct_of_sentences"] = (freq["n"] / freq["total_sentences"] * 100).round(1)

    category_totals = freq.groupby(["verb", "level", "category"])["n"].transform("sum")
    freq["pct_within_category"] = (freq["n"] / category_totals * 100).round(1)

    freq = freq.sort_values(["verb", "level", "category", "n"],
                             ascending=[True, True, True, False])
    return freq[["verb", "level", "category", "complement", "n",
                 "pct_of_sentences", "pct_within_category", "total_sentences"]]


def build_relativization_summary(raw_df):
    """
    Count and rate of relative-clause object recoveries (relativized=True) by
    verb x level. These are already folded into noun/proper_noun, but are also
    reported on their own: the A1 -> A2 -> B1+ trend in relativization is an
    independent syntactic signal that can be compared with the timing of
    semantic expansion for each verb.
    """
    total_sents = (raw_df.groupby(["verb", "level"]).size()
                   .reset_index(name="total_sentences"))
    rel_counts = (raw_df[raw_df["relativized"]]
                  .groupby(["verb", "level"]).size()
                  .reset_index(name="n_relativized"))
    summary = total_sents.merge(rel_counts, on=["verb", "level"], how="left")
    summary["n_relativized"] = summary["n_relativized"].fillna(0).astype(int)
    summary["pct_of_sentences"] = (summary["n_relativized"] / summary["total_sentences"] * 100).round(2)
    summary = summary.sort_values(["verb", "level"])
    return summary[["verb", "level", "n_relativized", "total_sentences", "pct_of_sentences"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="JEFLL_ten_verbs.csv")
    parser.add_argument("--verbs", nargs="*", default=["take", "make", "like"])
    parser.add_argument("--out_dir", default="output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    nlp = load_spacy()

    print("📥 Extracting and categorizing constructions...")
    raw_df = build_all_contexts(args.csv, args.verbs, nlp)

    print("📥 Attaching topic metadata (if present in CSV)...")
    raw_df = attach_topics(raw_df, args.csv)

    raw_df.to_csv(f"{args.out_dir}/constructions_all.csv", index=False)
    print(f"📁 Saved: {args.out_dir}/constructions_all.csv")

    print("\n🧮 Building category summary...")
    summary_df = build_category_summary(raw_df)
    summary_df.to_csv(f"{args.out_dir}/category_summary.csv", index=False)
    print(summary_df.to_string(index=False))
    print(f"📁 Saved: {args.out_dir}/category_summary.csv")

    print("\n🧮 Building noun collocate table...")
    noun_df = build_noun_collocates(raw_df)

    print("🧮 Building verbal complement table (gerund/infinitive)...")
    verbal_df = build_verbal_complements(raw_df)

    print("🧮 Building relativization summary...")
    relativization_df = build_relativization_summary(raw_df)
    relativization_df.to_csv(f"{args.out_dir}/relativization_summary.csv", index=False)

    xlsx_path = f"{args.out_dir}/extract_constructions.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        raw_df.to_excel(writer, sheet_name="raw_all", index=False)
        summary_df.to_excel(writer, sheet_name="category_summary", index=False)
        noun_df.to_excel(writer, sheet_name="noun_collocates", index=False)
        verbal_df.to_excel(writer, sheet_name="verbal_complements", index=False)
        relativization_df.to_excel(writer, sheet_name="relativization_summary", index=False)

        # Add filter dropdowns so every sheet can be filtered from the column
        # headers as soon as it is opened in Excel.
        for sheet_name, df in [
            ("raw_all", raw_df), ("category_summary", summary_df),
            ("noun_collocates", noun_df), ("verbal_complements", verbal_df),
            ("relativization_summary", relativization_df),
        ]:
            ws = writer.sheets[sheet_name]
            ws.auto_filter.ref = ws.dimensions
    print(f"\n💾 Saved: {xlsx_path}")


if __name__ == "__main__":
    main()
