import sys
import traceback

# まず、基本的なインポートを確認
print("=" * 50)
print("アプリケーションを起動しています...")
print("=" * 50)

try:
    print("\n[1/6] 基本ライブラリのインポート中...")
    import gradio as gr
    import pandas as pd
    import numpy as np
    import json
    import os
    from pathlib import Path
    print("✓ 基本ライブラリのインポート完了")
except Exception as e:
    print(f"✗ 基本ライブラリのインポートエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[2/6] 機械学習ライブラリのインポート中...")
    from transformers import BertForSequenceClassification, BertJapaneseTokenizer
    from catboost import CatBoostClassifier
    import joblib
    import torch
    from sklearn.feature_selection import SelectKBest, chi2
    print("✓ 機械学習ライブラリのインポート完了")
except Exception as e:
    print(f"✗ 機械学習ライブラリのインポートエラー: {e}")
    print("\n必要なパッケージをインストールしてください:")
    print("pip install -r requirements.txt")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[3/6] カスタムモジュールのインポート中...")
    from feature_extractor import extract_features, extract_features_with_pipelines
    from model_loader import load_all_models, load_models2
    from datetime import datetime
    from huggingface_hub import snapshot_download
    print("✓ カスタムモジュールのインポート完了")
except Exception as e:
    print(f"✗ カスタムモジュールのインポートエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# パス設定
BASE_DIR = Path(__file__).parent
MODEL_REPO_ID = "graduation-account/japaneseaa-models"
MODEL_CACHE_DIR = BASE_DIR / "hf_model_assets"

print("\nHugging Faceからモデルアセットを取得しています...")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    local_dir=str(MODEL_CACHE_DIR),
    local_dir_use_symlinks=False,
    resume_download=True,
)
print("✓ モデルアセットの取得完了")

MODELS_DIR = MODEL_CACHE_DIR / "models"
MODELS2_DIR = MODEL_CACHE_DIR / "models2"

CONFIG_PATH = MODELS_DIR / "ensemble_config.json"
CONFIG2_PATH = MODELS2_DIR / "ensemble_config.json"
DATASET_PATH = BASE_DIR / "dataset" / "minister_features.csv"
DATASET2_PATH = BASE_DIR / "dataset" / "JP_texts_with_dependency_features.csv"
LOG_DIR = BASE_DIR / "logs"

# ログディレクトリの作成
LOG_DIR.mkdir(exist_ok=True)

# 新しいデータセットの16人の著者ID（画像から取得）
NEW_AUTHOR_IDS = ["11077", "11031", "11040", "11005", "11042", "11056", "11027", "11009", 
                  "11011", "11028", "11036", "11016", "11047", "11037", "11068", "11075"]

try:
    print("\n[4/8] 既存システムの設定ファイルの読み込み中...")
    print(f"設定ファイルパス: {CONFIG_PATH}")
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {CONFIG_PATH}")
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print("✓ 既存システムの設定ファイルの読み込み完了")
except Exception as e:
    print(f"✗ 既存システムの設定ファイルの読み込みエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[5/8] 新規システムの設定ファイルの読み込み中...")
    print(f"設定ファイルパス: {CONFIG2_PATH}")
    if not CONFIG2_PATH.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {CONFIG2_PATH}")
    with open(CONFIG2_PATH, 'r', encoding='utf-8') as f:
        config2 = json.load(f)
    print("✓ 新規システムの設定ファイルの読み込み完了")
except Exception as e:
    print(f"✗ 新規システムの設定ファイルの読み込みエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[6/8] 既存データセットの読み込み中...")
    print(f"データセットパス: {DATASET_PATH}")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    available_authors_old = sorted(df['author'].unique().tolist())
    print(f"✓ 既存データセットの読み込み完了（著者数: {len(available_authors_old)}）")
except Exception as e:
    print(f"✗ 既存データセットの読み込みエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[7/8] 新規データセットの読み込み中...")
    print(f"データセットパス: {DATASET2_PATH}")
    if not DATASET2_PATH.exists():
        raise FileNotFoundError(f"データセットが見つかりません: {DATASET2_PATH}")
    df2 = pd.read_csv(DATASET2_PATH)
    # カラム名を確認（Author IDまたはauthor）
    author_column = config2['columns']['author'] if 'columns' in config2 and 'author' in config2['columns'] else 'Author ID'
    if author_column not in df2.columns:
        # もしAuthor IDが見つからない場合は、authorを試す
        author_column = 'author' if 'author' in df2.columns else None
        if author_column is None:
            raise KeyError(f"著者カラムが見つかりません。利用可能なカラム: {df2.columns.tolist()}")
    
    # 新しいデータセットから16人の著者のみを抽出
    # データセット内の著者IDを確認
    df2[author_column] = df2[author_column].astype(str)
    df2_filtered = df2[df2[author_column].isin(NEW_AUTHOR_IDS)]
    
    # 利用可能な著者IDを設定（データセットに存在するもののみ）
    # データセット内に実際に存在する著者IDのみを抽出
    existing_authors_in_dataset = sorted(df2[author_column].unique().tolist())
    available_authors_new = sorted([str(aid) for aid in NEW_AUTHOR_IDS if str(aid) in existing_authors_in_dataset])
    
    # データセット内に存在する著者IDを確認
    print(f"データセット内の著者ID数: {len(existing_authors_in_dataset)}")
    print(f"選択された著者ID数: {len(available_authors_new)}")
    print(f"利用可能な著者ID: {available_authors_new}")
    print(f"✓ 新規データセットの読み込み完了（著者数: {len(available_authors_new)}）")
except Exception as e:
    print(f"✗ 新規データセットの読み込みエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

try:
    print("\n[8/8] モデルの読み込み中...")
    print("※ 初回起動時は時間がかかる場合があります...")
    print("\n既存システムのモデルを読み込み中...")
    models = load_all_models(MODEL_CACHE_DIR, config)
    print("✓ 既存システムのモデルの読み込み完了")
    
    print("\n新規システムのモデルを読み込み中...")
    models2 = load_models2(MODEL_CACHE_DIR, config2)
    print("✓ 新規システムのモデルの読み込み完了")
except Exception as e:
    print(f"✗ モデルの読み込みエラー: {e}")
    print("\nモデルファイルが正しく配置されているか確認してください:")
    print(f"  - 既存BERTモデル: {BASE_DIR / 'models' / 'bert_minister'}")
    print(f"  - 新規BERTモデル: {BASE_DIR / 'models2' / 'bert_model'}")
    traceback.print_exc()
    sys.exit(1)

def prepare_features_for_model(features_dict):
    """特徴量辞書をモデル入力用の配列に変換（既存システム用）"""
    selected_features = config['preprocess']['selected_features']
    feature_vector = []
    
    for feat_name in selected_features:
        if feat_name in features_dict:
            feature_vector.append(features_dict[feat_name])
        else:
            feature_vector.append(0.0)
    
    return np.array(feature_vector).reshape(1, -1)

def prepare_features_for_model2(features_dict, feature_list):
    """特徴量辞書をモデル入力用の配列に変換（新規システム用）"""
    all_features = feature_list['all_features']
    feature_vector = []
    
    for feat_name in all_features:
        if feat_name in features_dict:
            feature_vector.append(features_dict[feat_name])
        else:
            feature_vector.append(0.0)
    
    return np.array(feature_vector).reshape(1, -1)

def write_log(text, selected_author, features, bert_probs, catboost_probs, ensemble_probs, result, has_unknown_author, author_group, classes):
    """Record analysis results to log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = LOG_DIR / f"similarity_log_{datetime.now().strftime('%Y%m%d')}.txt"
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Author Group: {author_group}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Input Text: {text}\n")
        f.write(f"Selected Author: {selected_author}\n")
        f.write("\n[Features]\n")
        # Record only key features
        key_features = [
            "Non-Independent Usage Rate",
            "Unique Dependencies",
            "Max Dependency Depth",
            "Avg Dependency Depth",
            "sentence_length"
        ]
        for feat_name in key_features:
            if feat_name in features:
                f.write(f"  {feat_name}: {features[feat_name]}\n")
        
        f.write("\n[BERT Prediction Probabilities]\n")
        for i, author_name in enumerate(classes):
            f.write(f"  {author_name}: {bert_probs[i]*100:.2f}%\n")
        
        f.write("\n[CatBoost Prediction Probabilities]\n")
        for i, author_name in enumerate(classes):
            f.write(f"  {author_name}: {catboost_probs[i]*100:.2f}%\n")
        
        f.write("\n[Ensemble Prediction Probabilities]\n")
        for i, author_name in enumerate(classes):
            f.write(f"  {author_name}: {ensemble_probs[i]*100:.2f}%\n")
        
        f.write(f"\n[Analysis Result]\n")
        if has_unknown_author:
            f.write("  ⚠ Possible Unknown Author (Max probability < 50%)\n")
        f.write(f"\n{result}\n")
        f.write("=" * 80 + "\n\n")

def predict_similarity(text, selected_author, author_group):
    """Calculate similarity between input text and selected author"""
    if not text or not text.strip():
        return "Error: Please enter text."
    
    if not author_group:
        return "Error: Please select an author group."
    
    # 著者グループに応じて利用可能な著者リストを選択
    if author_group == "Former PMs (3)":
        available_authors = available_authors_old
        use_new_model = False
        author_group_name = "Former PMs (3)"
    else:
        available_authors = available_authors_new
        use_new_model = True
        author_group_name = "NINJAL Corpus (16)"
    
    if not selected_author or selected_author not in available_authors:
        return f"Error: Please select a valid author from {author_group_name}."
    
    try:
        if use_new_model:
            # 新規システム（16人の著者）の場合
            # 1. 特徴量抽出（テキストパイプラインを使用）
            features = extract_features_with_pipelines(
                text, 
                models2['text_pipelines_manifest_path'],
                BASE_DIR / "models2"
            )
            
            # 2. モデル入力用に特徴量を準備
            feature_vector = prepare_features_for_model2(features, models2['feature_list'])
            
            # 3. BERT予測
            tokenizer = models2['tokenizer']
            bert_model = models2['bert_model']
            
            inputs = tokenizer(
                text,
                max_length=config2['bert']['tokenizer_max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                bert_outputs = bert_model(**inputs)
                bert_probs = torch.softmax(bert_outputs.logits, dim=-1).numpy()[0]
            
            # 4. CatBoost予測
            catboost_probs = models2['catboost_model'].predict_proba(feature_vector)[0]
            
            # 5. アンサンブル
            alpha = config2['alpha']
            ensemble_probs = alpha * bert_probs + (1 - alpha) * catboost_probs
            
            # 6. ラベルエンコーダーで著者IDに変換
            label_encoder = models2['label_encoder']
            # クラス名を取得（label_encoderから）
            classes = label_encoder.classes_.tolist()
            classes = [str(c) for c in classes]  # 文字列に変換
            
            # 選択した著者のインデックスを取得
            try:
                author_label = label_encoder.transform([selected_author])[0]
            except (ValueError, KeyError):
                # 著者IDが直接変換できない場合、classesのインデックスを使用
                if selected_author in classes:
                    author_label = classes.index(selected_author)
                else:
                    return f"Error: Author '{selected_author}' is not in the model's class list."
            
            num_authors = 16
        else:
            # 既存システム（3人の著者）の場合
            # 1. 特徴量抽出
            features = extract_features(text)
            
            # 2. モデル入力用に特徴量を準備
            feature_vector = prepare_features_for_model(features)
            
            # 3. chi_selectorで特徴量選択
            if models['chi_selector'] is not None:
                feature_vector = models['chi_selector'].transform(feature_vector)
            
            # 4. BERT予測
            tokenizer = models['tokenizer']
            bert_model = models['bert_model']
            
            inputs = tokenizer(
                text,
                max_length=config['tokenizer_max_length'],
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                bert_outputs = bert_model(**inputs)
                bert_probs = torch.softmax(bert_outputs.logits, dim=-1).numpy()[0]
            
            # 5. CatBoost予測
            catboost_probs = models['catboost_model'].predict_proba(feature_vector)[0]
            
            # 6. アンサンブル（alpha=0.7でBERT、0.3でCatBoost）
            alpha = config['alpha']
            ensemble_probs = alpha * bert_probs + (1 - alpha) * catboost_probs
            
            # 7. ラベルエンコーダーで著者名に変換
            label_encoder = models['label_encoder']
            classes = config['classes']
            
            # 選択した著者のインデックスを取得
            try:
                author_label = label_encoder.transform([selected_author])[0]
            except (ValueError, KeyError):
                # 著者名が直接変換できない場合、classesのインデックスを使用
                if selected_author in classes:
                    author_label = classes.index(selected_author)
                else:
                    return f"Error: Author '{selected_author}' is not in the model's class list."
            
            num_authors = 3
        
        # 類似度計算（選択した著者の確率）
        similarity_score = ensemble_probs[author_label] * 100
        
        # 最大確率とその著者を取得
        max_prob_idx = np.argmax(ensemble_probs)
        max_prob = ensemble_probs[max_prob_idx] * 100
        max_prob_author = classes[max_prob_idx]
        
        # 該当なしの判定（最大確率が50%未満の場合）
        UNKNOWN_THRESHOLD = 50.0
        has_unknown_author = max_prob < UNKNOWN_THRESHOLD
        
        # Format result
        result = f"Author Group: {author_group_name}\n"
        result += f"Selected Author: {selected_author}\n"
        result += f"Similarity Score: {similarity_score:.2f}%\n\n"
        
        # Display unknown author possibility
        if has_unknown_author:
            result += "⚠ Analysis Result: Possible Unknown Author\n"
            result += f"(Max Probability: {max_prob_author} {max_prob:.2f}% < Threshold {UNKNOWN_THRESHOLD}%)\n"
            result += f"The input text may not match any of the trained authors ({num_authors} authors).\n\n"
        else:
            result += f"✓ Analysis Result: Highest similarity with \"{max_prob_author}\" ({max_prob:.2f}%)\n\n"
        
        result += f"Prediction Probabilities for All Authors ({num_authors} authors):\n"
        result += "-" * 40 + "\n"
        
        # Sort by probability (highest first)
        prob_with_index = [(ensemble_probs[i] * 100, i, classes[i]) for i in range(len(classes))]
        prob_with_index.sort(reverse=True)
        
        for prob, idx, author_name in prob_with_index:
            markers = []
            if str(author_name) == str(selected_author):
                markers.append("← Selected")
            if idx == max_prob_idx:
                markers.append("★ Highest Probability")
            marker_str = " ".join(markers) if markers else ""
            result += f"{author_name}: {prob:.2f}% {marker_str}\n"
        
        # ログ出力
        try:
            write_log(text, selected_author, features, bert_probs, catboost_probs, ensemble_probs, result, has_unknown_author, author_group_name, classes)
        except Exception as log_error:
            # Ignore log output errors (still return result)
            print(f"Warning: Error occurred during log output: {log_error}")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred: {str(e)}\n\n"
        error_msg += f"Details:\n{traceback.format_exc()}"
        return error_msg

# 著者グループ選択に応じて著者リストを更新する関数
def update_author_list(author_group):
    """Update author list based on selected author group"""
    try:
        if author_group == "Former PMs (3)":
            choices = available_authors_old
        else:
            choices = available_authors_new
        
        if choices and len(choices) > 0:
            # Gradioのバージョンに応じて適切な形式で返す
            try:
                # 新しいバージョンではgr.update()を使用
                return gr.update(choices=choices, value=choices[0], interactive=True)
            except (AttributeError, TypeError):
                # 古いバージョンでは辞書を返す
                return {
                    "choices": choices,
                    "value": choices[0],
                    "interactive": True
                }
        else:
            # 空のリストの場合はエラーメッセージを表示
            error_msg = f"No authors available for {author_group}"
            print(f"Warning: {error_msg}")
            try:
                return gr.update(choices=[error_msg], value=None, interactive=False)
            except (AttributeError, TypeError):
                return {
                    "choices": [error_msg],
                    "value": None,
                    "interactive": False
                }
    except Exception as e:
        print(f"Error in update_author_list: {e}")
        import traceback
        traceback.print_exc()
        try:
            return gr.update(choices=[f"Error: {str(e)}"], value=None, interactive=False)
        except (AttributeError, TypeError):
            return {
                "choices": [f"Error: {str(e)}"],
                "value": None,
                "interactive": False
            }

# Gradioインターフェース
print("\n" + "=" * 50)
print("Gradioインターフェースを構築中...")
print("=" * 50)

with gr.Blocks(title="Author Similarity Analysis System") as demo:
    gr.Markdown("# Author Similarity Analysis System")
    gr.Markdown("Enter Japanese text and analyze its similarity with the selected author.")
    
    with gr.Row():
        with gr.Column():
            author_group_radio = gr.Radio(
                choices=["Former PMs (3)", "NINJAL Corpus (16)"],
                label="Author Group Selection",
                value="Former PMs (3)",
                info="Please select an author group to use"
            )
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Enter Japanese text here...",
                lines=5
            )
            author_dropdown = gr.Dropdown(
                choices=available_authors_old if available_authors_old and len(available_authors_old) > 0 else [],
                label="Target Author",
                value=available_authors_old[0] if available_authors_old and len(available_authors_old) > 0 else None,
                interactive=True
            )
            predict_btn = gr.Button("Calculate Similarity", variant="primary")
        
        with gr.Column():
            output_text = gr.Textbox(
                label="Result",
                lines=15,
                interactive=False
            )
    
    # 著者グループ選択時に著者リストを更新
    author_group_radio.change(
        fn=update_author_list,
        inputs=author_group_radio,
        outputs=author_dropdown
    )
    
    # 予測ボタンのクリックイベント
    predict_btn.click(
        fn=predict_similarity,
        inputs=[text_input, author_dropdown, author_group_radio],
        outputs=output_text
    )
    
    gr.Markdown("### Available Authors")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Former PMs (3)")
            gr.Markdown(", ".join(available_authors_old))
        with gr.Column():
            gr.Markdown("#### NINJAL Corpus (16)")
            gr.Markdown(", ".join(available_authors_new))

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Starting Gradio server...")
    print("Please access the URL displayed after startup.")
    print("=" * 50 + "\n")
    
    try:
        demo.launch()
    except Exception as e:
        print(f"\n✗ Gradio server startup error: {e}")
        traceback.print_exc()
        sys.exit(1)
