import sys
import traceback
from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import uvicorn
import socket
import argparse

# まず、基本的なインポートを確認
print("=" * 50)
print("アプリケーションを起動しています...")
print("=" * 50)

try:
    print("\n[1/6] 基本ライブラリのインポート中...")
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
    # feature_extractorは遅延読み込みなので、インポート時にはエラーが発生しない
    from feature_extractor import extract_features, extract_features_with_pipelines
    from model_loader import load_all_models, load_models2
    from datetime import datetime
    print("✓ カスタムモジュールのインポート完了")
except ImportError as e:
    print(f"✗ カスタムモジュールのインポートエラー: {e}")
    print("\n注意: GinZAモデルの読み込みエラーが発生する可能性があります。")
    print("以下のコマンドでモデルを再インストールしてください:")
    print("  py -m spacy download ja_ginza_electra")
    print("\nまたは、PyTorchをアップグレードしてください:")
    print("  pip install torch>=2.1.0")
    print("\nアプリケーションは起動しますが、特徴量抽出時にエラーが発生する可能性があります。")
    traceback.print_exc()
    # インポートエラーでも続行（実際の使用時にエラーが発生する）
except Exception as e:
    print(f"✗ カスタムモジュールのインポートエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# Webスクレイピングライブラリのインポート（遅延読み込み用）
_requests_available = False
_bs4_available = False

try:
    print("\n[3.5/6] Webスクレイピングライブラリのインポート中...")
    import requests
    from bs4 import BeautifulSoup
    _requests_available = True
    _bs4_available = True
    print("✓ Webスクレイピングライブラリのインポート完了")
except Exception as e:
    print(f"✗ Webスクレイピングライブラリのインポートエラー: {e}")
    print("\n必要なパッケージをインストールしてください:")
    print("pip install requests beautifulsoup4 lxml")
    print("※ URL機能は使用できませんが、アプリケーションは起動します。")
    traceback.print_exc()
    # エラーでも続行（URL機能は使用不可になる）
except ImportError as e:
    print(f"✗ カスタムモジュールのインポートエラー: {e}")
    print("\n注意: GinZAモデルの読み込みエラーが発生する可能性があります。")
    print("以下のコマンドでモデルを再インストールしてください:")
    print("  py -m spacy download ja_ginza_electra")
    print("\nまたは、PyTorchをアップグレードしてください:")
    print("  pip install torch>=2.1.0")
    print("\nアプリケーションは起動しますが、特徴量抽出時にエラーが発生する可能性があります。")
    traceback.print_exc()
    # インポートエラーでも続行（実際の使用時にエラーが発生する）
except Exception as e:
    print(f"✗ カスタムモジュールのインポートエラー: {e}")
    traceback.print_exc()
    sys.exit(1)

# numpy型をPythonネイティブ型に変換する関数
def convert_numpy_types(obj):
    """numpy型をPythonネイティブ型に変換（再帰的）"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy_types(item) for item in obj)
    else:
        return obj

# FastAPIアプリケーションの初期化
app = FastAPI(title="Author Similarity Analysis System")

# 静的ファイルとテンプレートの設定
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

from jinja2 import Environment, FileSystemLoader
templates = Environment(loader=FileSystemLoader(str(BASE_DIR / "templates")))

# パス設定
CONFIG_PATH = BASE_DIR / "models" / "ensemble_config.json"
CONFIG2_PATH = BASE_DIR / "models2" / "ensemble_config.json"
DATASET_PATH = BASE_DIR / "dataset" / "minister_features.csv"
DATASET2_PATH = BASE_DIR / "dataset" / "JP_texts_with_dependency_features.csv"
LOG_DIR = BASE_DIR / "logs"

# ログディレクトリの作成
LOG_DIR.mkdir(exist_ok=True)

# 新しいデータセットの16人の著者ID（画像から取得）
NEW_AUTHOR_IDS = ["11077", "11031", "11040", "11005", "11042", "11056", "11027", "11009", 
                  "11011", "11028", "11036", "11016", "11047", "11037", "11068", "11075"]

# グローバル変数
config = None
config2 = None
available_authors_old = []
available_authors_new = []
models = None
models2 = None

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
    models = load_all_models(BASE_DIR, config)
    print("✓ 既存システムのモデルの読み込み完了")
    
    print("\n新規システムのモデルを読み込み中...")
    models2 = load_models2(BASE_DIR, config2)
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
        # Record all features
        for feat_name in sorted(features.keys()):
            feat_value = features[feat_name]
            # Handle different types of feature values
            if isinstance(feat_value, (int, float)):
                f.write(f"  {feat_name}: {feat_value}\n")
            elif isinstance(feat_value, (list, tuple)):
                f.write(f"  {feat_name}: {feat_value}\n")
            elif hasattr(feat_value, 'most_common'):  # Counter object
                # Show top 10 most common items for Counter objects
                top_items = feat_value.most_common(10)
                if top_items:
                    f.write(f"  {feat_name}: {dict(top_items)}\n")
                else:
                    f.write(f"  {feat_name}: {{}}\n")
            elif isinstance(feat_value, dict):
                f.write(f"  {feat_name}: {feat_value}\n")
            elif isinstance(feat_value, set):
                f.write(f"  {feat_name}: {sorted(feat_value)}\n")
            else:
                f.write(f"  {feat_name}: {str(feat_value)}\n")
        
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
        raise ValueError("Please enter text.")
    
    if not author_group:
        raise ValueError("Please select an author group.")
    
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
        raise ValueError(f"Please select a valid author from {author_group_name}.")
    
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
                    raise ValueError(f"Author '{selected_author}' is not in the model's class list.")
            
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
                    raise ValueError(f"Author '{selected_author}' is not in the model's class list.")
            
            num_authors = 3
        
        # 類似度計算（選択した著者の確率）
        similarity_score = float(ensemble_probs[author_label] * 100)  # numpy型をfloatに変換
        
        # 最大確率とその著者を取得
        max_prob_idx = int(np.argmax(ensemble_probs))  # numpy型をintに変換
        max_prob = float(ensemble_probs[max_prob_idx] * 100)  # numpy型をfloatに変換
        max_prob_author = classes[max_prob_idx]
        
        # 該当なしの判定（最大確率が50%未満の場合）
        UNKNOWN_THRESHOLD = 50.0
        has_unknown_author = bool(max_prob < UNKNOWN_THRESHOLD)  # numpy型をboolに変換
        
        # 結果を辞書形式で返す
        result = {
            "author_group": author_group_name,
            "selected_author": selected_author,
            "similarity_score": round(similarity_score, 2),
            "max_prob_author": max_prob_author,
            "max_prob": round(max_prob, 2),
            "has_unknown_author": has_unknown_author,
            "unknown_threshold": UNKNOWN_THRESHOLD,
            "num_authors": num_authors,
            "predictions": []
        }
        
        # Sort by probability (highest first)
        prob_with_index = [(float(ensemble_probs[i] * 100), int(i), classes[i]) for i in range(len(classes))]  # numpy型を変換
        prob_with_index.sort(reverse=True)
        
        for prob, idx, author_name in prob_with_index:
            is_selected = str(author_name) == str(selected_author)
            is_max = idx == max_prob_idx
            result["predictions"].append({
                "author": author_name,
                "probability": round(prob, 2),
                "is_selected": bool(is_selected),  # 念のためboolに変換
                "is_max": bool(is_max)  # 念のためboolに変換
            })
        
        # ログ出力
        try:
            result_text = f"Author Group: {author_group_name}\n"
            result_text += f"Selected Author: {selected_author}\n"
            result_text += f"Similarity Score: {similarity_score:.2f}%\n\n"
            
            if has_unknown_author:
                result_text += "⚠ Analysis Result: Possible Unknown Author\n"
                result_text += f"(Max Probability: {max_prob_author} {max_prob:.2f}% < Threshold {UNKNOWN_THRESHOLD}%)\n"
                result_text += f"The input text may not match any of the trained authors ({num_authors} authors).\n\n"
            else:
                result_text += f"✓ Analysis Result: Highest similarity with \"{max_prob_author}\" ({max_prob:.2f}%)\n\n"
            
            result_text += f"Prediction Probabilities for All Authors ({num_authors} authors):\n"
            result_text += "-" * 40 + "\n"
            
            for prob, idx, author_name in prob_with_index:
                markers = []
                if str(author_name) == str(selected_author):
                    markers.append("← Selected")
                if idx == max_prob_idx:
                    markers.append("★ Highest Probability")
                marker_str = " ".join(markers) if markers else ""
                result_text += f"{author_name}: {prob:.2f}% {marker_str}\n"
            
            write_log(text, selected_author, features, bert_probs, catboost_probs, ensemble_probs, result_text, has_unknown_author, author_group_name, classes)
        except Exception as log_error:
            # Ignore log output errors (still return result)
            print(f"Warning: Error occurred during log output: {log_error}")
        
        return result
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred: {str(e)}\n\n"
        error_msg += f"Details:\n{traceback.format_exc()}"
        raise RuntimeError(error_msg)

def split_text_by_punctuation(text):
    """テキストを「。」で分割"""
    sentences = [s.strip() + '。' for s in text.split('。') if s.strip()]
    # 最後の文が「。」で終わっていない場合は追加しない
    if text and not text.endswith('。'):
        if sentences:
            sentences[-1] = sentences[-1].rstrip('。')
    return sentences

def fetch_text_from_url(url):
    """URLからテキストを取得"""
    # 必要なライブラリをインポート（常にインポートを試みる）
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError as e:
        raise ValueError(f"必要なライブラリ（requests, beautifulsoup4）がインストールされていません。エラー: {str(e)}\npip install requests beautifulsoup4 lxml を実行してください。")
    
    # Twitter/XのURLの場合はSeleniumを使用
    if 'twitter.com' in url or 'x.com' in url:
        try:
            from selenium import webdriver
            from selenium.webdriver.chrome.service import Service
            from selenium.webdriver.chrome.options import Options
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
            from webdriver_manager.chrome import ChromeDriverManager
            import time
        except ImportError:
            raise ValueError("X（Twitter）のリンクからテキストを取得するには、Seleniumが必要です。\npip install selenium webdriver-manager を実行してください。")
        
        try:
            # Chromeオプションを設定（ヘッドレスモード）
            chrome_options = Options()
            chrome_options.add_argument('--headless')  # ブラウザを表示しない
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
            
            # WebDriverを起動
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
            
            try:
                # Xのページを開く
                driver.get(url)
                
                # コンテンツが読み込まれるまで待機（最大30秒）
                wait = WebDriverWait(driver, 30)
                
                # ツイートのテキストを取得（複数のセレクタを試行）
                tweet_text = None
                selectors = [
                    'article[data-testid="tweet"] div[data-testid="tweetText"]',
                    'article div[lang]',
                    'article span[lang]',
                    '[data-testid="tweetText"]'
                ]
                
                for selector in selectors:
                    try:
                        element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                        tweet_text = element.text
                        if tweet_text and len(tweet_text.strip()) > 0:
                            break
                    except:
                        continue
                
                # 見つからない場合は少し待ってから再試行
                if not tweet_text:
                    time.sleep(3)
                    try:
                        article = driver.find_element(By.CSS_SELECTOR, 'article[data-testid="tweet"]')
                        tweet_text = article.text
                    except:
                        pass
                
                if not tweet_text or len(tweet_text.strip()) == 0:
                    raise ValueError("ツイートのテキストを取得できませんでした。リンクが正しいか確認してください。")
                
                return tweet_text.strip()
                
            finally:
                driver.quit()
                
        except Exception as e:
            raise ValueError(f"X（Twitter）からのテキスト取得中にエラーが発生しました: {str(e)}")
    
    # 通常のURLの処理（既存のコード）
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding
        
        soup = BeautifulSoup(response.text, 'lxml')
        
        # スクリプトとスタイルタグを削除
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 主要なコンテンツタグからテキストを抽出
        text_parts = []
        
        # articleタグを優先
        article = soup.find('article')
        if article:
            text_parts.append(article.get_text(separator=' ', strip=True))
        
        # mainタグ
        main = soup.find('main')
        if main:
            text_parts.append(main.get_text(separator=' ', strip=True))
        
        # pタグからテキストを抽出
        paragraphs = soup.find_all('p')
        if paragraphs:
            para_text = ' '.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
            if para_text:
                text_parts.append(para_text)
        
        # どれも見つからない場合はbody全体から
        if not text_parts:
            body = soup.find('body')
            if body:
                text_parts.append(body.get_text(separator=' ', strip=True))
        
        # テキストを結合して整形
        full_text = ' '.join(text_parts)
        # 余分な空白を削除
        full_text = ' '.join(full_text.split())
        
        if not full_text:
            raise ValueError("テキストを取得できませんでした。")
        
        return full_text
    except requests.exceptions.Timeout:
        raise ValueError("URLへの接続がタイムアウトしました。")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"URLへの接続エラー: {str(e)}")
    except Exception as e:
        raise ValueError(f"テキストの取得中にエラーが発生しました: {str(e)}")

def predict_similarity_for_all_authors(text, author_group):
    """テキストに対して全著者の類似度を計算（selected_authorなし）"""
    if not text or not text.strip():
        raise ValueError("Please enter text.")
    
    if not author_group:
        raise ValueError("Please select an author group.")
    
    # 著者グループに応じて利用可能な著者リストを選択
    if author_group == "Former PMs (3)":
        available_authors = available_authors_old
        use_new_model = False
        author_group_name = "Former PMs (3)"
    else:
        available_authors = available_authors_new
        use_new_model = True
        author_group_name = "NINJAL Corpus (16)"
    
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
            classes = label_encoder.classes_.tolist()
            classes = [str(c) for c in classes]  # 文字列に変換
            
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
            
            # 6. アンサンブル
            alpha = config['alpha']
            ensemble_probs = alpha * bert_probs + (1 - alpha) * catboost_probs
            
            # 7. ラベルエンコーダーで著者名に変換
            label_encoder = models['label_encoder']
            classes = config['classes']
            
            num_authors = 3
        
        # 全著者に対する類似度スコアを計算
        all_results = {}
        for i, author_name in enumerate(classes):
            similarity_score = float(ensemble_probs[i] * 100)
            all_results[str(author_name)] = {
                "similarity_score": round(similarity_score, 2),
                "max_prob_author": str(author_name),
                "max_prob": round(similarity_score, 2)
            }
        
        # 最大確率の著者を特定
        max_author = max(all_results.items(), key=lambda x: x[1]["similarity_score"])
        
        return {
            "author_group": author_group_name,
            "all_results": all_results,
            "max_similarity_author": max_author[0],
            "max_similarity_score": max_author[1]["similarity_score"]
        }
        
    except Exception as e:
        import traceback
        error_msg = f"An error occurred: {str(e)}\n\n"
        error_msg += f"Details:\n{traceback.format_exc()}"
        raise RuntimeError(error_msg)

def predict_similarity_detailed(text, author_group):
    """句読点ごとに細かく類似度を分析"""
    if not text or not text.strip():
        raise ValueError("Please enter text.")
    
    if not author_group:
        raise ValueError("Please select an author group.")
    
    # テキストを句読点で分割
    sentences = split_text_by_punctuation(text)
    
    if not sentences:
        raise ValueError("有効な文章が見つかりませんでした。")
    
    # 全体の分析結果
    whole_result = None
    try:
        # 全体に対して最大確率の著者を取得
        whole_analysis = predict_similarity_for_all_authors(text, author_group)
        whole_result = {
            "max_author": whole_analysis["max_similarity_author"],
            "max_score": whole_analysis["max_similarity_score"]
        }
    except Exception as e:
        pass
    
    # 各文の分析結果
    sentence_results = []
    for idx, sentence in enumerate(sentences):
        try:
            sentence_analysis = predict_similarity_for_all_authors(sentence, author_group)
            sentence_results.append({
                "sentence_index": idx + 1,
                "sentence": sentence,
                "max_author": sentence_analysis["max_similarity_author"],
                "max_score": sentence_analysis["max_similarity_score"],
                "all_scores": sentence_analysis["all_results"]
            })
        except Exception as e:
            sentence_results.append({
                "sentence_index": idx + 1,
                "sentence": sentence,
                "error": str(e)
            })
    
    # 最も類似度が高い著者を集計
    author_counts = {}
    for result in sentence_results:
        if "max_author" in result:
            author = result["max_author"]
            if author not in author_counts:
                author_counts[author] = {"count": 0, "total_score": 0.0}
            author_counts[author]["count"] += 1
            author_counts[author]["total_score"] += result["max_score"]
    
    # 最も多く出現した著者を特定
    most_common_author = None
    if author_counts:
        most_common_author = max(author_counts.items(), key=lambda x: x[1]["count"])
    
    # 通常の結果表示形式を生成（全体テキストの分析結果から）
    standard_result = None
    if whole_result:
        try:
            # 全体テキストの分析結果を取得
            whole_analysis = predict_similarity_for_all_authors(text, author_group)
            max_author = whole_analysis["max_similarity_author"]
            max_score = whole_analysis["max_similarity_score"]
            all_results = whole_analysis["all_results"]
            
            # 著者グループ名を取得
            if author_group == "Former PMs (3)":
                author_group_name = "Former PMs (3)"
                num_authors = 3
            else:
                author_group_name = "NINJAL Corpus (16)"
                num_authors = 16
            
            # 予測結果をソート
            predictions = []
            for author, scores in sorted(all_results.items(), key=lambda x: x[1]["similarity_score"], reverse=True):
                predictions.append({
                    "author": author,
                    "probability": scores["similarity_score"],
                    "is_selected": author == max_author,
                    "is_max": author == max_author
                })
            
            # 通常の結果表示形式を生成
            UNKNOWN_THRESHOLD = 50.0
            standard_result = {
                "author_group": author_group_name,
                "selected_author": max_author,  # 最大確率の著者を選択
                "similarity_score": round(max_score, 2),
                "max_prob_author": max_author,
                "max_prob": round(max_score, 2),
                "has_unknown_author": max_score < UNKNOWN_THRESHOLD,
                "unknown_threshold": UNKNOWN_THRESHOLD,
                "num_authors": num_authors,
                "predictions": predictions
            }
        except Exception as e:
            pass
    
    return {
        "whole_result": whole_result,
        "sentence_results": sentence_results,
        "most_common_author": most_common_author[0] if most_common_author else None,
        "author_counts": {k: {"count": v["count"], "avg_score": v["total_score"] / v["count"]} 
                         for k, v in author_counts.items()} if author_counts else {},
        "standard_result": standard_result  # 通常の結果表示形式を追加
    }

# APIエンドポイント
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """メインページを返す"""
    template = templates.get_template("index.html")
    html_content = template.render(
        available_authors_old=available_authors_old,
        available_authors_new=available_authors_new
    )
    return HTMLResponse(content=html_content)

@app.get("/api/authors")
async def get_authors(author_group: str):
    """著者リストを取得"""
    if author_group == "Former PMs (3)":
        return {"authors": available_authors_old}
    elif author_group == "NINJAL Corpus (16)":
        return {"authors": available_authors_new}
    else:
        raise HTTPException(status_code=400, detail="Invalid author group")

@app.post("/api/predict")
async def predict(request: Request):
    """類似度を予測"""
    try:
        data = await request.json()
        text = data.get("text", "")
        selected_author = data.get("selected_author", "")
        author_group = data.get("author_group", "")
        
        result = predict_similarity(text, selected_author, author_group)
        # numpy型をPythonネイティブ型に変換してからJSONResponseに渡す
        result = convert_numpy_types(result)
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/fetch-url")
async def fetch_url(request: Request):
    """URLからテキストを取得"""
    try:
        data = await request.json()
        url = data.get("url", "")
        
        if not url:
            raise ValueError("URLを入力してください。")
        
        text = fetch_text_from_url(url)
        return JSONResponse(content={"text": text, "success": True})
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": str(e), "text": ""}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e), "text": ""}
        )

@app.post("/api/predict-detailed")
async def predict_detailed(request: Request):
    """細かい類似度分析を実行"""
    try:
        data = await request.json()
        text = data.get("text", "")
        author_group = data.get("author_group", "")
        analysis_mode = data.get("analysis_mode", "both")  # "whole", "detailed", "both"
        
        if not text or not text.strip():
            raise ValueError("テキストを入力してください。")
        
        if not author_group:
            raise ValueError("著者グループを選択してください。")
        
        result = {}
        
        # 全体の分析
        if analysis_mode in ["whole", "both"]:
            try:
                whole_analysis = predict_similarity_for_all_authors(text, author_group)
                result["whole"] = convert_numpy_types(whole_analysis)
            except Exception as e:
                result["whole"] = {"error": str(e)}
        
        # 細かい分析
        if analysis_mode in ["detailed", "both"]:
            try:
                detailed_analysis = predict_similarity_detailed(text, author_group)
                result["detailed"] = convert_numpy_types(detailed_analysis)
            except Exception as e:
                result["detailed"] = {"error": str(e)}
        
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def is_port_in_use(host, port):
    """ポートが使用中かどうかをチェック"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True

def find_available_port(host, start_port, max_attempts=10):
    """利用可能なポートを見つける"""
    for i in range(max_attempts):
        port = start_port + i
        if not is_port_in_use(host, port):
            return port
    return None

def kill_process_on_port(port):
    """指定されたポートを使用しているプロセスを終了する（Windows用）"""
    import subprocess
    try:
        # netstatでポートを使用しているプロセスを検索
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.split('\n')
        for line in lines:
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        subprocess.run(["taskkill", "/F", "/PID", pid], check=True, capture_output=True)
                        print(f"ポート{port}を使用していたプロセス（PID: {pid}）を終了しました。")
                        return True
                    except subprocess.CalledProcessError:
                        pass
    except Exception as e:
        print(f"プロセスの終了に失敗しました: {e}")
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="著者類似度分析システム")
    parser.add_argument("--port", type=int, default=8000, help="使用するポート番号（デフォルト: 8000）")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="ホストアドレス（デフォルト: 127.0.0.1）")
    parser.add_argument("--kill-existing", action="store_true", help="既存のプロセスを終了する")
    parser.add_argument("--auto-port", action="store_true", help="ポートが使用中の場合、自動的に別のポートを使用する")
    
    args = parser.parse_args()
    
    host = args.host
    port = args.port
    
    # ポートが使用中の場合の処理
    if is_port_in_use(host, port):
        print(f"\n⚠ 警告: ポート{port}は既に使用中です。")
        
        if args.kill_existing:
            print(f"既存のプロセスを終了しようとしています...")
            if kill_process_on_port(port):
                print(f"プロセスを終了しました。ポート{port}を使用します。")
            else:
                print(f"プロセスを終了できませんでした。")
                if args.auto_port:
                    print("利用可能なポートを検索しています...")
                    new_port = find_available_port(host, port)
                    if new_port:
                        port = new_port
                        print(f"ポート{new_port}を使用します。")
                    else:
                        print("❌ エラー: 利用可能なポートが見つかりませんでした。")
                        sys.exit(1)
                else:
                    print("❌ エラー: ポートが使用中です。以下のオプションを試してください:")
                    print("  --kill-existing: 既存のプロセスを終了する")
                    print("  --auto-port: 自動的に別のポートを使用する")
                    sys.exit(1)
        elif args.auto_port:
            print(f"利用可能なポートを検索しています...")
            new_port = find_available_port(host, port)
            if new_port:
                port = new_port
                print(f"ポート{new_port}を使用します。")
            else:
                print("❌ エラー: 利用可能なポートが見つかりませんでした。")
                sys.exit(1)
        else:
            print("❌ エラー: ポートが使用中です。以下のオプションを試してください:")
            print("  --kill-existing: 既存のプロセスを終了する")
            print("  --auto-port: 自動的に別のポートを使用する")
            sys.exit(1)
    
    print("\n" + "=" * 50)
    print("Starting FastAPI server...")
    print(f"Please access http://{host}:{port} in your browser.")
    print("=" * 50 + "\n")
    
    uvicorn.run(app, host=host, port=port)

