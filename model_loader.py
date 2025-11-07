import json
from pathlib import Path
from transformers import BertForSequenceClassification, BertJapaneseTokenizer
from catboost import CatBoostClassifier
import joblib
import torch

def load_all_models(base_dir, config):
    """すべてのモデルを読み込む"""
    models = {}
    
    # パスの調整（ローカルパス用）
    bert_path = base_dir / "models" / "bert_minister"
    catboost_path = base_dir / "models" / "catboost_minister.cbm"
    label_encoder_path = base_dir / "models" / "preprocess" / "label_encoder.joblib"
    chi_selector_path = base_dir / "models" / "preprocess" / "chi_selector.joblib"
    
    print(f"BERTモデルを読み込み中: {bert_path}")
    # BERTモデルとトークナイザー
    # safetensors形式のモデルファイルを自動検出して読み込む
    try:
        models['bert_model'] = BertForSequenceClassification.from_pretrained(
            str(bert_path),
            torch_dtype=torch.float32,
            local_files_only=True
        )
    except Exception as e:
        # safetensorsが読み込めない場合、use_safetensorsをFalseにしてみる
        print(f"通常の方法で読み込み失敗: {e}")
        print("代替方法を試行します...")
        models['bert_model'] = BertForSequenceClassification.from_pretrained(
            str(bert_path),
            torch_dtype=torch.float32,
            local_files_only=True,
            use_safetensors=False
        )
    models['bert_model'].eval()  # 評価モード
    
    models['tokenizer'] = BertJapaneseTokenizer.from_pretrained(str(bert_path))
    
    print(f"CatBoostモデルを読み込み中: {catboost_path}")
    # CatBoostモデル
    models['catboost_model'] = CatBoostClassifier()
    models['catboost_model'].load_model(str(catboost_path))
    
    print(f"Label Encoderを読み込み中: {label_encoder_path}")
    # Label Encoder
    models['label_encoder'] = joblib.load(label_encoder_path)
    
    print(f"Chi Selectorを読み込み中: {chi_selector_path}")
    # Chi Selector
    models['chi_selector'] = joblib.load(chi_selector_path)
    
    print("すべてのモデルの読み込みが完了しました。")
    return models

def load_models2(base_dir, config):
    """models2用のモデルを読み込む"""
    models = {}
    
    # パスの調整
    bert_path = base_dir / "models2" / "bert_model"
    catboost_path = base_dir / "models2" / "catboost_model.cbm"
    label_encoder_path = base_dir / "models2" / "preprocess" / "label_encoder.joblib"
    feature_list_path = base_dir / "models2" / "preprocess" / "feature_list.json"
    text_pipelines_manifest_path = base_dir / "models2" / "preprocess" / "text_pipelines_manifest.json"
    
    print(f"BERTモデルを読み込み中: {bert_path}")
    # BERTモデルとトークナイザー
    try:
        models['bert_model'] = BertForSequenceClassification.from_pretrained(
            str(bert_path),
            torch_dtype=torch.float32,
            local_files_only=True
        )
    except Exception as e:
        print(f"通常の方法で読み込み失敗: {e}")
        print("代替方法を試行します...")
        models['bert_model'] = BertForSequenceClassification.from_pretrained(
            str(bert_path),
            torch_dtype=torch.float32,
            local_files_only=True,
            use_safetensors=False
        )
    models['bert_model'].eval()  # 評価モード
    
    models['tokenizer'] = BertJapaneseTokenizer.from_pretrained(str(bert_path))
    
    print(f"CatBoostモデルを読み込み中: {catboost_path}")
    # CatBoostモデル
    models['catboost_model'] = CatBoostClassifier()
    models['catboost_model'].load_model(str(catboost_path))
    
    print(f"Label Encoderを読み込み中: {label_encoder_path}")
    # Label Encoder
    models['label_encoder'] = joblib.load(label_encoder_path)
    
    print(f"Feature Listを読み込み中: {feature_list_path}")
    # Feature List
    with open(feature_list_path, 'r', encoding='utf-8') as f:
        models['feature_list'] = json.load(f)
    
    print(f"Text Pipelines Manifestを読み込み中: {text_pipelines_manifest_path}")
    # Text Pipelines Manifest
    models['text_pipelines_manifest_path'] = str(text_pipelines_manifest_path)
    
    print("models2のすべてのモデルの読み込みが完了しました。")
    return models


