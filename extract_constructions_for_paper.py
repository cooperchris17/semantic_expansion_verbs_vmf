# -*- coding: utf-8 -*-
"""
extract_constructions.py
=========================
対象動詞（take / make / like）が取る補文・目的語のパターンを、
名詞目的語だけでなく動名詞・不定詞・節・等位接続・受動態主語・
述語形容詞・句動詞まで含めて分類し、頻度表を Excel に出力する。

collocate_freq.py（旧版）との違い:
  旧版は dobj/attr/pobj の NOUN/PROPN しか見ておらず、それ以外の文は
  .dropna(subset=["noun"]) で無言で捨てられていた。
  本スクリプトは同じ動詞トークンの子ノードを見て、以下すべてを分類する:

    noun / proper_noun     : 名詞目的語（dobj, attr, pobj, dative）
    coordinated_noun       : 上記に等位接続された追加の名詞（"rice and bread"）
    pronoun                : 代名詞目的語（"like it"）
    gerund                 : 動名詞補文（xcomp, VBG）（"like swimming"）
    infinitive              : 不定詞補文（xcomp, VB）（"like to swim"）
    clausal                : 節補文（ccomp）（"like that we can..."）
    predicate_adjective    : 形容詞補語（acomp）（"make sure"）
    passive_subject        : 受動態の主語（nsubjpass）（"bread was made"）
    phrasal_only           : 名詞・節等は無いが不変化詞のみ（"take off"）
    none                   : 上記いずれも無し（自動詞的・省略）

  各動詞トークンにつき「最も情報量の高い」1カテゴリを主カテゴリとして
  記録する（優先順位は CONSTRUCTION_PRIORITY を参照）。等位接続された
  追加の名詞は主カテゴリとは別に、名詞コロケート頻度表にのみ加算する
  （カテゴリ集計を二重計上しないため）。

  1文につき、対象動詞の最初の出現のみを見る（旧版と同じ制約）。

出力（すべて output/ 以下）:
  constructions_all.csv      : 動詞出現ごとの生データ（文・カテゴリ・コロケート）
  category_summary.csv       : 動詞×レベル×カテゴリの頻度・割合
  extract_constructions.xlsx : 上記をまとめた Excel ブック
      - raw_all              : 生データ全件
      - category_summary     : カテゴリ別頻度・割合（ピボット）
      - noun_collocates      : 名詞コロケート頻度表（noun + proper_noun + coordinated_noun）
      - verbal_complements   : 動名詞・不定詞補文の頻度表

使い方:
  python extract_constructions.py --csv JEFLL_ten_verbs.csv --verbs take make like
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from shared_utils import load_data_with_ids, load_spacy

TARGET_LEVELS = ["A1", "A2", "B1+"]

# 主カテゴリを決めるときの優先順位（上にあるほど優先）。
# phrasal_only / none はここに含めない（candidates が空のときのフォールバックとして別処理）。
CONSTRUCTION_PRIORITY = [
    "noun", "proper_noun", "pronoun", "coordinated_noun",
    "gerund", "infinitive", "clausal",
    "predicate_adjective", "passive_subject",
]

NOUN_CATEGORIES = {"noun", "proper_noun", "coordinated_noun"}
VERBAL_CATEGORIES = {"gerund", "infinitive"}


def extract_constructions(sentences, verb, nlp):
    """
    sentences（"文" または (文, 作文ID) タプルのリスト）から、対象動詞の
    出現ごとに構文カテゴリを分類した dict のリストを返す。
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

            # 関係節の目的語ギャップ処理:
            # token（対象動詞）自身が関係節の述語（dep_=="relcl"）で、
            # かつ節内に主語（nsubj/nsubjpass）が既にある場合、
            # 欠けている項は目的語である可能性が高い。
            # この場合 token.head が先行詞（=真の意味的目的語）なので、
            # "that/which"（またはゼロ関係代名詞）を pronoun として拾うのではなく
            # 先行詞そのものを noun/proper_noun として回収する。
            # 例: "the bread that I make" → make.dep_=="relcl", make.head=="bread"
            #     "the man who makes bread" → makes.dep_=="relcl" だが主語ギャップ
            #     （who が nsubj）なので対象外。bread は通常の dobj で拾われる。
            if token.dep_ == "relcl" and token.head.pos_ in {"NOUN", "PROPN"}:
                # 主語が「関係代名詞自身」(who/that, tag_=="WP"/"WDT") なら主語関係節
                # （例: "the man WHO makes bread" — who が主語）なので対象外。
                subject_is_wh = any(
                    c.dep_ in {"nsubj", "nsubjpass"} and c.tag_ in {"WP", "WDT"}
                    for c in token.children
                )
                # 通常の（wh でない）主語があるか（例: "the bread that I make" の I）
                has_real_subject = any(
                    c.dep_ in {"nsubj", "nsubjpass"} and c.tag_ not in {"WP", "WDT"}
                    for c in token.children
                )
                # 目的語スロットが既に実質名詞で埋まっていないか
                # （埋まっていれば関係節の欠落項は目的語ではない＝別の関係、例: 副詞的関係節）
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
                        # relativized=True の場合、この PRON は "that/which" 等の
                        # 関係代名詞（先行詞は上ですでに回収済み）である可能性が高いので、
                        # 別途 pronoun として二重計上しない。
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

            # 主カテゴリ = 優先順位に沿って最初に見つかったもの
            primary = None
            for cat in CONSTRUCTION_PRIORITY:
                match = next((c for c in candidates if c[0] == cat), None)
                if match:
                    primary = match
                    break

            if primary is None:
                primary = ("phrasal_only", None) if particle else ("none", None)

            category, collocate = primary

            # 主カテゴリ以外の等位接続名詞（名詞コロケート表にのみ使う）
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
            break  # 1文につき最初の1箇所だけ

    return contexts


def build_all_contexts(csv_path, verbs, nlp):
    """全動詞・全レベルについて extract_constructions を実行し、長形式 DataFrame を返す。"""
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
    元の CSV に topic 列があれば (file_id, sentence) -> topic のルックアップ用
    DataFrame を返す。無ければ None（呼び出し側で attach_topics がスキップする）。
    file_id は load_data_with_ids 側と型を揃えるため文字列に統一しておく
    （int64 vs object の不一致で merge が落ちるのを防ぐ）。

    注意: 元 CSV は (verb, sentence) ごとに1行なので、1文に対象動詞が複数
    含まれる場合（例: "I take money and make food"）は同じ (file_id, sentence)
    が複数行に渡って重複する。重複したまま merge すると行が意図せず増える
    （fan-out）ため、ここで (file_id, sentence) を一意化してから返す。
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.lower().str.strip()
    if "topic" not in df.columns:
        print("ℹ️ CSV に 'topic' 列が見つからないため、トピック情報はスキップします。")
        return None
    topics = df[["file", "sentence", "topic"]].copy()
    topics.columns = ["file_id", "sentence", "topic"]
    topics["file_id"] = topics["file_id"].astype(str)
    topics["sentence"] = topics["sentence"].astype(str).str.strip()

    # 完全重複行（同じ file_id/sentence/topic）はそのまま集約
    topics = topics.drop_duplicates(subset=["file_id", "sentence", "topic"])

    # 同じ (file_id, sentence) で topic 値が食い違う行が残っていれば警告し、
    # 最初の1件だけ残す（merge を必ず 1 対 1 にするため）
    dup_mask = topics.duplicated(subset=["file_id", "sentence"], keep=False)
    if dup_mask.any():
        n_conflicting = topics.loc[dup_mask, ["file_id", "sentence"]].drop_duplicates().shape[0]
        print(f"⚠️ {n_conflicting} 件の (file_id, sentence) で topic 値が食い違って"
              f"います。最初の値を採用します。")
        topics = topics.drop_duplicates(subset=["file_id", "sentence"], keep="first")

    return topics


def attach_topics(raw_df, csv_path):
    """
    raw_df に topic 列を付与する。(file_id, sentence) で結合するため、
    raw_df 側の file_id も文字列化してから merge する。
    結合できなかった行数、および merge 前後で行数が変わっていないかを
    表示し、想定外の欠落・重複増加に気づけるようにする。
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
        print(f"⚠️ merge で行数が {n_before} → {n_after} に変化しました。"
              f"topic の結合キーが一意でない可能性があります。")

    n_missing = merged["topic"].isna().sum()
    if n_missing:
        print(f"⚠️ {n_missing} / {len(merged)} 行で topic が結合できませんでした "
              f"（file_id/sentence の不一致の可能性）。")
    return merged


def build_category_summary(raw_df):
    """動詞×レベル×カテゴリの頻度・割合表を作る。"""
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
    名詞コロケート頻度表（noun + proper_noun + 主カテゴリの coordinated_noun + extra_nouns）。

    2種類の割合を付与する:
      pct_of_sentences     : その動詞×レベルの全文数に対する割合
                              （category_summary の pct と同じ分母。シート間で直接比較可能）
      pct_within_category  : 名詞コロケート全体に対する割合
                              （そのレベルで見つかった名詞の中での相対頻度＝ランキングの重み）
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
    動名詞・不定詞補文の頻度表（"like playing", "like to play" 等）。

    pct_of_sentences     : その動詞×レベルの全文数に対する割合
    pct_within_category  : 同じ補文タイプ（gerund または infinitive）内での相対頻度
                            （gerund と infinitive は別の合計に対して計算する）
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
    動詞×レベルごとの関係節目的語ギャップ回収（relativized=True）の件数・割合。
    noun/proper_noun には既に merge 済みだが、独立した現象として集計しておく
    （関係節化率の A1→A2→B1+ 推移は、動詞ごとの意味拡張タイミングと
    比較できる独立した統語的シグナルになりうる）。
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

        # 各シートにフィルタ用のドロップダウンを付けておく（Excel で開いた時点で
        # そのまま列見出しをクリックしてフィルタできるように）
        for sheet_name, df in [
            ("raw_all", raw_df), ("category_summary", summary_df),
            ("noun_collocates", noun_df), ("verbal_complements", verbal_df),
            ("relativization_summary", relativization_df),
        ]:
            ws = writer.sheets[sheet_name]
            ws.auto_filter.ref = ws.dimensions
    print(f"\n💾 Saved: {xlsx_path}")

    # 参考: 動詞×レベルごとの「名詞目的語カバー率」を旧版と比較できるよう表示
    print("\n📊 Coverage check (category = noun/proper_noun/coordinated_noun only, "
          "i.e. what the OLD collocate_freq.py would have kept):")
    old_style = summary_df[summary_df["category"].isin(NOUN_CATEGORIES)]
    old_totals = old_style.groupby(["verb", "level"])[["n", "total"]].sum()
    old_totals["pct_kept_by_old_script"] = (old_totals["n"] / old_totals["total"] * 100).round(1)
    print(old_totals.to_string())

    # 参考: 関係節の目的語ギャップから先行詞を回収した件数・割合
    # （"the bread that I make" 型。noun/proper_noun に merge 済みだが、
    #   独立した現象としても集計済み。詳細は output/relativization_summary.csv
    #   および Excel の relativization_summary シートを参照。）
    print("\n📊 Relative-clause object recoveries (folded into noun/proper_noun, "
          "also saved as its own sheet/CSV):")
    print(relativization_df.to_string(index=False))


if __name__ == "__main__":
    main()