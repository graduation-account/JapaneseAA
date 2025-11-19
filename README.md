# 著者類似度判定システム / Author Similarity Analysis System

日本語文章を入力し、訓練済みBERT+CatBoostアンサンブルモデルを使用して、選択した著者との類似度を判定するWebアプリケーションです。

This is a web application that analyzes Japanese text and calculates similarity with selected authors using pre-trained BERT+CatBoost ensemble models.

---

## データセットとモデルの対応関係 / Dataset and Model Correspondence

### 日本語 / Japanese

システムは2つの著者グループをサポートしています：

1. **Former PMs (3) - 元総理大臣3人**
   - **データセット**: `dataset/minister_features.csv`
   - **モデル**: `models/` フォルダ内
     - BERTモデル: `models/bert_minister/`
     - CatBoostモデル: `models/catboost_minister.cbm`
     - 前処理モデル: `models/preprocess/`
   - **著者**: 安倍晋三、岸田文雄、石破茂

2. **NINJAL Corpus (16) - NINJAL作文コーパス16人**
   - **データセット**: `dataset/JP_texts_with_dependency_features.csv`
   - **モデル**: `models2/` フォルダ内
     - BERTモデル: `models2/bert_model/`
     - CatBoostモデル: `models2/catboost_model.cbm`
     - 前処理モデル: `models2/preprocess/`
   - **著者ID**: 11005, 11009, 11011, 11016, 11027, 11028, 11031, 11036, 11037, 11040, 11042, 11047, 11056, 11068, 11075, 11077

### English

The system supports two author groups:

1. **Former PMs (3) - Former Prime Ministers**
   - **Dataset**: `dataset/minister_features.csv`
   - **Models**: Located in `models/` folder
     - BERT model: `models/bert_minister/`
     - CatBoost model: `models/catboost_minister.cbm`
     - Preprocessing models: `models/preprocess/`
   - **Authors**: Shinzo Abe, Fumio Kishida, Shigeru Ishiba

2. **NINJAL Corpus (16) - NINJAL Writing Corpus**
   - **Dataset**: `dataset/JP_texts_with_dependency_features.csv`
   - **Models**: Located in `models2/` folder
     - BERT model: `models2/bert_model/`
     - CatBoost model: `models2/catboost_model.cbm`
     - Preprocessing models: `models2/preprocess/`
   - **Author IDs**: 11005, 11009, 11011, 11016, 11027, 11028, 11031, 11036, 11037, 11040, 11042, 11047, 11056, 11068, 11075, 11077

---

## セットアップ / Setup

### 日本語 / Japanese

1. **依存関係のインストール**:
```bash
pip install -r requirements.txt
```

2. **GinZAモデルのダウンロード**:
```bash
python -m spacy download ja_ginza_electra
```

3. **アプリケーションの起動**:
```bash
python app.py
```

ブラウザで http://127.0.0.1:8000 にアクセスして使用できます。

### English

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download GinZA model**:
```bash
python -m spacy download ja_ginza_electra
```

3. **Launch the application**:
```bash
python app.py
```

Access http://127.0.0.1:8000 in your browser to use the application.

### Git LFSファイルのトラブルシューティング / Git LFS File Troubleshooting

**日本語 / Japanese**

GitHubからリポジトリをクローンまたはプルした後、アプリケーションを起動しようとするとエラーが発生する場合があります。これは、Git LFSで管理されている大きなファイル（モデルファイルなど）がポインタファイルのままになっているためです。

以下のコマンドを実行して、実際のファイルを取得してください：

```bash
git lfs install
git lfs fetch --all
git lfs checkout
```

または、提供されているバッチファイルを実行してください：

```bash
fix_lfs_files.bat
```

**English**

After cloning or pulling the repository from GitHub, you may encounter errors when trying to launch the application. This is because large files managed by Git LFS (such as model files) remain as pointer files.

Run the following commands to retrieve the actual files:

```bash
git lfs install
git lfs fetch --all
git lfs checkout
```

Or run the provided batch file:

```bash
fix_lfs_files.bat
```

---

## 使用方法 / How to Use

### 日本語 / Japanese

1. **著者グループの選択**
   - 「著者グループ選択」から使用する著者グループを選択します
   - 「Former PMs (3)」: 元総理大臣3人のデータセット
   - 「NINJAL Corpus (16)」: NINJAL作文コーパスから抽出した16人のデータセット

2. **テキストの入力**
   - 「入力テキスト」欄に分析したい日本語の文章を入力します

3. **著者の選択**
   - 「対象著者」ドロップダウンから比較したい著者を選択します
   - 選択した著者グループに応じて、利用可能な著者リストが自動的に更新されます

4. **類似度の計算**
   - 「類似度を計算」ボタンをクリックします
   - または、テキスト入力欄で Ctrl+Enter（Macでは Cmd+Enter）を押すと実行できます

5. **結果の確認**
   - 右側の「分析結果」パネルに以下の情報が表示されます：
     - 選択した著者との類似度スコア（パーセンテージ）
     - すべての著者に対する予測確率（プログレスバー付き）
     - 分析結果（最大確率が50%未満の場合は「未知の著者の可能性」が表示されます）

### English

1. **Select Author Group**
   - Choose the author group from "著者グループ選択" (Author Group Selection)
   - "Former PMs (3)": Dataset of 3 former Japanese Prime Ministers
   - "NINJAL Corpus (16)": Dataset of 16 authors extracted from NINJAL Writing Corpus

2. **Enter Text**
   - Input the Japanese text you want to analyze in the "入力テキスト" (Input Text) field

3. **Select Author**
   - Choose the author to compare from the "対象著者" (Target Author) dropdown
   - The available author list is automatically updated based on the selected author group

4. **Calculate Similarity**
   - Click the "類似度を計算" (Calculate Similarity) button
   - Or press Ctrl+Enter (Cmd+Enter on Mac) in the text input field

5. **View Results**
   - The "分析結果" (Analysis Result) panel on the right displays:
     - Similarity score with the selected author (percentage)
     - Prediction probabilities for all authors (with progress bars)
     - Analysis result (if the maximum probability is below 50%, a "Possible Unknown Author" warning is displayed)

---

## ファイル構成 / File Structure

### 日本語 / Japanese

```
AA system_v4/
├── app.py                          # メインアプリケーション（FastAPI）
├── feature_extractor.py            # 特徴量抽出モジュール
├── model_loader.py                 # モデルローダー
├── requirements.txt                # 依存関係
├── README.md                       # このファイル
│
├── templates/                      # HTMLテンプレート
│   └── index.html                  # メインページ
│
├── static/                         # 静的ファイル
│   ├── style.css                   # スタイルシート
│   └── script.js                   # JavaScript
│
├── models/                         # 元総理大臣3人用モデル
│   ├── bert_minister/              # BERTモデル
│   ├── catboost_minister.cbm       # CatBoostモデル
│   ├── ensemble_config.json        # 設定ファイル
│   └── preprocess/                 # 前処理モデル
│       ├── label_encoder.joblib
│       └── chi_selector.joblib
│
├── models2/                        # NINJALコーパス16人用モデル
│   ├── bert_model/                 # BERTモデル
│   ├── catboost_model.cbm          # CatBoostモデル
│   ├── ensemble_config.json        # 設定ファイル
│   └── preprocess/                 # 前処理モデル（テキストパイプライン含む）
│       ├── label_encoder.joblib
│       ├── feature_list.json
│       ├── text_pipelines_manifest.json
│       └── pipe_*.joblib           # テキストパイプライン
│
├── dataset/
│   ├── minister_features.csv       # 元総理大臣3人のデータセット
│   └── JP_texts_with_dependency_features.csv  # NINJALコーパス16人のデータセット
│
└── logs/                           # ログファイル
    └── similarity_log_YYYYMMDD.txt
```

### English

```
AA system_v4/
├── app.py                          # Main application (FastAPI)
├── feature_extractor.py            # Feature extraction module
├── model_loader.py                 # Model loader
├── requirements.txt                # Dependencies
├── README.md                       # This file
│
├── templates/                      # HTML templates
│   └── index.html                  # Main page
│
├── static/                         # Static files
│   ├── style.css                   # Stylesheet
│   └── script.js                   # JavaScript
│
├── models/                         # Models for 3 Former PMs
│   ├── bert_minister/              # BERT model
│   ├── catboost_minister.cbm       # CatBoost model
│   ├── ensemble_config.json        # Configuration file
│   └── preprocess/                 # Preprocessing models
│       ├── label_encoder.joblib
│       └── chi_selector.joblib
│
├── models2/                        # Models for 16 NINJAL Corpus authors
│   ├── bert_model/                 # BERT model
│   ├── catboost_model.cbm          # CatBoost model
│   ├── ensemble_config.json        # Configuration file
│   └── preprocess/                 # Preprocessing models (including text pipelines)
│       ├── label_encoder.joblib
│       ├── feature_list.json
│       ├── text_pipelines_manifest.json
│       └── pipe_*.joblib           # Text pipelines
│
├── dataset/
│   ├── minister_features.csv       # Dataset for 3 Former PMs
│   └── JP_texts_with_dependency_features.csv  # Dataset for 16 NINJAL Corpus authors
│
└── logs/                           # Log files
    └── similarity_log_YYYYMMDD.txt
```

---

## システムの動作 / How the System Works

### 日本語 / Japanese

1. **特徴量抽出**
   - **Former PMs (3)**: GinZAを使用して日本語テキストから形態素解析を行い、基本的な特徴量を抽出します
   - **NINJAL Corpus (16)**: GinZAを使用した形態素解析に加えて、テキストパイプラインを使用してSVD特徴量も生成します

2. **BERT予測**
   - 訓練済みBERTモデルでテキストを分類します
   - Former PMs用: `cl-tohoku/bert-base-japanese-v2`（最大長128トークン）
   - NINJAL Corpus用: `cl-tohoku/bert-large-japanese-v2`（最大長256トークン）

3. **CatBoost予測**
   - 抽出した特徴量を使用してCatBoostモデルで分類します

4. **アンサンブル**
   - BERT（70%）とCatBoost（30%）の予測を組み合わせます（alpha=0.7）

5. **類似度表示**
   - 選択した著者との類似度をパーセンテージで表示します
   - すべての著者に対する予測確率を降順で表示します（プログレスバー付き）
   - 最大確率が50%未満の場合は「未知の著者の可能性」を警告します

6. **ログ出力**
   - 分析結果は `logs/similarity_log_YYYYMMDD.txt` に記録されます

### English

1. **Feature Extraction**
   - **Former PMs (3)**: Uses GinZA for morphological analysis of Japanese text and extracts basic features
   - **NINJAL Corpus (16)**: In addition to morphological analysis with GinZA, generates SVD features using text pipelines

2. **BERT Prediction**
   - Classifies text using pre-trained BERT models
   - For Former PMs: `cl-tohoku/bert-base-japanese-v2` (max length 128 tokens)
   - For NINJAL Corpus: `cl-tohoku/bert-large-japanese-v2` (max length 256 tokens)

3. **CatBoost Prediction**
   - Classifies using CatBoost model with extracted features

4. **Ensemble**
   - Combines BERT (70%) and CatBoost (30%) predictions (alpha=0.7)

5. **Similarity Display**
   - Displays similarity with the selected author as a percentage
   - Shows prediction probabilities for all authors in descending order (with progress bars)
   - Warns about "Possible Unknown Author" if maximum probability is below 50%

6. **Logging**
   - Analysis results are recorded in `logs/similarity_log_YYYYMMDD.txt`

---

## APIエンドポイント / API Endpoints

### 日本語 / Japanese

このアプリケーションはRESTful APIも提供しています：

- `GET /`: メインページ（HTML）
- `GET /api/authors?author_group={group}`: 著者リストを取得
- `POST /api/predict`: 類似度を予測
  - リクエストボディ: `{"text": "...", "selected_author": "...", "author_group": "..."}`
  - レスポンス: JSON形式の予測結果

### English

This application also provides RESTful API endpoints:

- `GET /`: Main page (HTML)
- `GET /api/authors?author_group={group}`: Get author list
- `POST /api/predict`: Predict similarity
  - Request body: `{"text": "...", "selected_author": "...", "author_group": "..."}`
  - Response: Prediction result in JSON format

---

## 注意事項 / Notes

### 日本語 / Japanese

- 初回起動時はモデルの読み込みに時間がかかる場合があります
- テキストは日本語で入力してください
- ログファイルは日付ごとに自動的に作成されます
- NINJAL Corpus (16)のモデルは、テキストパイプラインを使用するため、より多くの特徴量を処理します
- このシステムはFastAPIとカスタムHTML/CSS/JavaScriptを使用しており、Gradioベースのv3とは異なるUIを提供します

### English

- Model loading may take time on first launch
- Please input text in Japanese
- Log files are automatically created for each date
- The NINJAL Corpus (16) model processes more features as it uses text pipelines
- This system uses FastAPI with custom HTML/CSS/JavaScript, providing a different UI from the Gradio-based v3

---

## 技術スタック / Technical Stack

- **Python 3.9+**
- **FastAPI**: Modern web framework for building APIs
- **Uvicorn**: ASGI server
- **Jinja2**: Template engine
- **Transformers**: BERT models
- **CatBoost**: Gradient boosting
- **spaCy + GinZA**: Japanese morphological analysis
- **PyTorch**: Deep learning framework
- **scikit-learn**: Machine learning utilities
- **pandas, numpy**: Data processing
- **HTML/CSS/JavaScript**: Frontend UI

---

## ライセンス / License

このプロジェクトのライセンス情報は含まれていません。使用前に確認してください。

License information for this project is not included. Please check before use.

