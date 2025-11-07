import spacy
import pandas as pd
from collections import Counter

# GinZA-Electraのロード
nlp = spacy.load("ja_ginza_electra")

# 特徴量抽出関数
def extract_features(text):
    doc = nlp(text)

    total_tokens = len(doc)

    # 1. 非自立語使用率
    non_independent = sum(1 for token in doc if token.pos_ in ["AUX", "PART", "ADP", "SCONJ"])
    non_indep_rate = non_independent / total_tokens if total_tokens > 0 else 0

    # 2. 品詞のビグラム
    pos_bigrams = Counter([
        f"{doc[i].pos_}_{doc[i+1].pos_}"
        for i in range(len(doc) - 1)
    ])

    # 3. 助詞のビグラム
    particle_bigrams = Counter([
        f"{doc[i].lemma_}_{doc[i+1].lemma_}"
        for i in range(len(doc) - 1)
        if doc[i].pos_ == "ADP" and doc[i+1].pos_ == "ADP"
    ])

    # 4. 句読点直前のトークン
    punctuation_preceding = Counter([
        doc[i-1].lemma_
        for i in range(1, len(doc))
        if doc[i].text in ["、", "。"]
    ])

    # 5. 依存関係に関する特徴量
    dependency_path_patterns = []
    unique_dependencies = set()
    forward_dependencies = 0
    backward_dependencies = 0
    subj_verb_order = 0
    obj_verb_order = 0
    adj_noun_order = 0
    sentence_length = total_tokens

    max_depth = 0
    avg_depth = 0
    total_depth = 0

    for token in doc:
        # 依存関係パスの収集
        dependency_path_patterns.append(f"{token.dep_} ({token.head.text} -> {token.text})")

        # ユニークな依存関係タイプを収集
        unique_dependencies.add(token.dep_)

        # 順方向、逆方向の依存関係数
        if token.dep_ != "punct" and token.head.dep_ != "punct":
            if token.i < token.head.i:
                forward_dependencies += 1
            else:
                backward_dependencies += 1

        # 主語-動詞、目的語-動詞、形容詞-名詞の語順関係の計算
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
            subj_verb_order += 1
        if token.dep_ == "obj" and token.head.pos_ == "VERB":
            obj_verb_order += 1
        if token.pos_ == "ADJ" and token.head.pos_ == "NOUN":
            adj_noun_order += 1

        # 依存関係ツリーの深さを計算
        depth = 0
        current_token = token
        while current_token.head != current_token:
            current_token = current_token.head
            depth += 1
        max_depth = max(max_depth, depth)
        total_depth += depth

    avg_depth = total_depth / total_tokens if total_tokens > 0 else 0

    return {
        "Non-Independent Usage Rate": non_indep_rate,
        "POS Bigrams": pos_bigrams,
        "Particle Bigrams": particle_bigrams,
        "Punctuation Preceding Tokens": punctuation_preceding,
        "Dependency Path Patterns": Counter(dependency_path_patterns),
        "Unique Dependencies": len(unique_dependencies),
        "Max Dependency Depth": max_depth,
        "Avg Dependency Depth": avg_depth,
        "Forward Dependencies": forward_dependencies,
        "Backward Dependencies": backward_dependencies,
        "sentence_length": sentence_length,
        "subj_verb_order_ratio": subj_verb_order / sentence_length if sentence_length > 0 else 0,
        "obj_verb_order_ratio": obj_verb_order / sentence_length if sentence_length > 0 else 0,
        "adj_noun_order_ratio": adj_noun_order / sentence_length if sentence_length > 0 else 0
    }

# データの読み込み（ファイルパスを適宜変更）
df = pd.read_csv("/content/drive/MyDrive/Database/minister_dataset.csv")  # あなたのファイルパスに合わせて

# 各テキストに特徴量を適用
features = df["text"].apply(extract_features)

# 結果を保存するためのリスト
output_rows = []

# 各発言に対する特徴量を抽出して保存
for idx, feature_dict in enumerate(features):
    feature_dict['author'] = df.iloc[idx]['author']
    feature_dict['post date'] = df.iloc[idx]['post date']
    feature_dict['post time'] = df.iloc[idx]['post time']
    feature_dict['text'] = df.iloc[idx]['text']
    output_rows.append(feature_dict)

# 結果をDataFrameに変換
output_df = pd.DataFrame(output_rows)

# CSVファイルに保存
output_df.to_csv("/content/drive/MyDrive/minister_features_extracted.csv", index=False)

print("特徴量の抽出が完了しました。結果はCSVファイルに保存されました。")