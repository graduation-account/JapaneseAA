import spacy
from collections import Counter
import json
import joblib
from pathlib import Path
import numpy as np

# GinZA-Electraのロード（遅延読み込み）
_nlp = None

def get_nlp():
    """GinZAモデルを遅延読み込みする関数"""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("ja_ginza_electra")
        except OSError as e:
            raise ImportError(
                f"GinZAモデルが見つかりません。以下でインストールしてください:\n"
                f"pip install -U ginza transformers spacy\n"
                f"python -m spacy download ja_ginza_electra\n"
                f"エラー詳細: {e}"
            )
        except ValueError as e:
            # モデルのデシリアライゼーションエラーの場合
            error_msg = str(e)
            if "E149" in error_msg or "deserializing" in error_msg.lower():
                raise ImportError(
                    f"GinZAモデルの読み込み中にエラーが発生しました（モデルのバージョン不一致の可能性）。\n"
                    f"以下の手順でモデルを再インストールしてください:\n\n"
                    f"1. 既存のモデルを削除:\n"
                    f"   python -m spacy info ja_ginza_electra\n"
                    f"   （モデルのパスを確認後、手動で削除）\n\n"
                    f"2. パッケージを再インストール:\n"
                    f"   pip uninstall -y ja-ginza-electra\n"
                    f"   pip install -U ginza transformers spacy\n"
                    f"   python -m spacy download ja_ginza_electra\n\n"
                    f"エラー詳細: {e}"
                )
            else:
                raise ImportError(
                    f"GinZAモデルの読み込み中にエラーが発生しました。\n"
                    f"モデルを再インストールしてください:\n"
                    f"python -m spacy download ja_ginza_electra\n"
                    f"エラー詳細: {e}"
                )
        except Exception as e:
            raise ImportError(
                f"GinZAモデルの読み込み中にエラーが発生しました。\n"
                f"以下の手順でモデルを再インストールしてください:\n\n"
                f"1. pip uninstall -y ja-ginza-electra\n"
                f"2. pip install -U ginza transformers spacy\n"
                f"3. python -m spacy download ja_ginza_electra\n\n"
                f"エラー詳細: {e}"
            )
    return _nlp

def extract_features(text):
    """テキストから特徴量を抽出する関数（extract_features.pyをベース）"""
    nlp = get_nlp()
    doc = nlp(text)
    
    total_tokens = len(doc)
    
    # 1. 非自立語使用率
    non_independent = sum(1 for token in doc if token.pos_ in ["AUX", "PART", "ADP", "SCONJ"])
    non_indep_rate = non_independent / total_tokens if total_tokens > 0 else 0
    
    # 2. 品詞のビグラム（数値化は不要、モデルでは使用しない）
    pos_bigrams = Counter([
        f"{doc[i].pos_}_{doc[i+1].pos_}"
        for i in range(len(doc) - 1)
    ])
    
    # 3. 助詞のビグラム（数値化は不要）
    particle_bigrams = Counter([
        f"{doc[i].lemma_}_{doc[i+1].lemma_}"
        for i in range(len(doc) - 1)
        if doc[i].pos_ == "ADP" and doc[i+1].pos_ == "ADP"
    ])
    
    # 4. 句読点直前のトークン（数値化は不要）
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

def extract_features_with_pipelines(text, text_pipelines_manifest_path, base_dir):
    """テキストから特徴量を抽出し、テキストパイプラインを使用してSVD特徴量を生成"""
    nlp = get_nlp()
    doc = nlp(text)
    
    total_tokens = len(doc)
    
    # 1. 非自立語使用率
    non_independent = sum(1 for token in doc if token.pos_ in ["AUX", "PART", "ADP", "SCONJ"])
    non_indep_rate = non_independent / total_tokens if total_tokens > 0 else 0
    
    # 2. 品詞のビグラム
    pos_bigrams = [
        f"{doc[i].pos_}_{doc[i+1].pos_}"
        for i in range(len(doc) - 1)
    ]
    
    # 3. 助詞のビグラム
    particle_bigrams = [
        f"{doc[i].lemma_}_{doc[i+1].lemma_}"
        for i in range(len(doc) - 1)
        if doc[i].pos_ == "ADP" and doc[i+1].pos_ == "ADP"
    ]
    
    # 4. 句読点直前のトークン
    punctuation_preceding = [
        doc[i-1].lemma_
        for i in range(1, len(doc))
        if doc[i].text in ["、", "。"]
    ]
    
    # 5. 依存関係に関する特徴量
    dependency_path_patterns = []
    dependency_pairs = []
    dependency_types = []
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
    total_root_distance = 0
    
    for token in doc:
        # 依存関係パスの収集
        dependency_path_patterns.append(f"{token.dep_} ({token.head.text} -> {token.text})")
        
        # 依存関係ペアの収集
        if token.head != token:
            dependency_pairs.append(f"{token.head.text}_{token.text}")
        
        # 依存関係タイプの収集
        dependency_types.append(token.dep_)
        
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
        root_distance = 0
        current_token = token
        while current_token.head != current_token:
            current_token = current_token.head
            depth += 1
            root_distance += 1
        max_depth = max(max_depth, depth)
        total_depth += depth
        total_root_distance += root_distance
    
    avg_depth = total_depth / total_tokens if total_tokens > 0 else 0
    avg_root_distance = total_root_distance / total_tokens if total_tokens > 0 else 0
    
    # テキストパイプラインを読み込んでSVD特徴量を生成
    base_path = Path(base_dir)
    with open(text_pipelines_manifest_path, 'r', encoding='utf-8') as f:
        pipelines_manifest = json.load(f)
    
    svd_features = {}
    
    # テキストデータを準備
    text_data = {
        "POS Bigrams": " ".join(pos_bigrams),
        "Particle Bigrams": " ".join(particle_bigrams),
        "Punctuation Preceding Tokens": " ".join(punctuation_preceding),
        "dependency_path_patterns": " ".join(dependency_path_patterns),
        "Dependency Pairs": " ".join(dependency_pairs),
        "Dependency Types": " ".join(dependency_types)
    }
    
    # 各パイプラインを適用
    for pipeline_name, pipeline_info in pipelines_manifest['text_pipelines'].items():
        # パイプラインファイル名を取得（絶対パスからファイル名のみを抽出）
        pipeline_filename = Path(pipeline_info['pipeline_path']).name
        # preprocessディレクトリ内のパイプラインファイルを探す
        pipeline_path = base_path / "preprocess" / pipeline_filename
        
        if pipeline_path.exists():
            try:
                pipeline = joblib.load(pipeline_path)
                text_input = text_data.get(pipeline_name, "")
                if text_input:
                    svd_output = pipeline.transform([text_input])[0]
                    for i, col_name in enumerate(pipeline_info['svd_columns']):
                        svd_features[col_name] = float(svd_output[i]) if i < len(svd_output) else 0.0
                else:
                    # テキストがない場合は0で埋める
                    for col_name in pipeline_info['svd_columns']:
                        svd_features[col_name] = 0.0
            except Exception as e:
                print(f"Warning: Error loading pipeline {pipeline_name}: {e}")
                # エラーが発生した場合は0で埋める
                for col_name in pipeline_info['svd_columns']:
                    svd_features[col_name] = 0.0
        else:
            # パイプラインが見つからない場合は0で埋める
            print(f"Warning: Pipeline file not found: {pipeline_path}")
            for col_name in pipeline_info['svd_columns']:
                svd_features[col_name] = 0.0
    
    # 数値特徴量とSVD特徴量を組み合わせて返す
    return {
        "adj_noun_order_ratio": adj_noun_order / sentence_length if sentence_length > 0 else 0,
        "subj_verb_order_ratio": subj_verb_order / sentence_length if sentence_length > 0 else 0,
        "obj_verb_order_ratio": obj_verb_order / sentence_length if sentence_length > 0 else 0,
        "avg_root_distance": avg_root_distance,
        "Forward Dependencies": forward_dependencies,
        "Backward Dependencies": backward_dependencies,
        "Token Count": total_tokens,
        "sentence_length": sentence_length,
        "Max Dependency Depth": max_depth,
        "Avg Dependency Depth": avg_depth,
        "Non-Independent Usage Rate": non_indep_rate,
        **svd_features  # SVD特徴量を追加
    }



