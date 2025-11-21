# 著者類似度判定システム v5 / Author Similarity Analysis System v5

日本語文章を入力し、訓練済みBERT+CatBoostアンサンブルモデルを使用して、選択した著者との類似度を判定するWebアプリケーションです。v5では、URLからのテキスト取得機能と句読点ごとの詳細分析機能が追加されました。

This is a web application that analyzes Japanese text and calculates similarity with selected authors using pre-trained BERT+CatBoost ensemble models. Version 5 adds URL text fetching and sentence-by-sentence detailed analysis features.

---

## 新機能 / New Features (v5)

### 日本語 / Japanese

- **URLからのテキスト取得**: 任意のURLを入力して、リンク先の文章を自動で取得し、類似度を計算できます
- **詳細分析機能**: 句読点で区切った各文章ごとに類似度を計算し、最も類似度の高い著者を表示します
- **分析モード選択**: 「全体テキストのみ」または「詳細分析（句読点ごと）」を選択できます

### English

- **URL Text Fetching**: Enter any URL to automatically fetch text from the link and calculate similarity
- **Detailed Analysis**: Calculate similarity for each sentence separated by punctuation and display the author with the highest similarity
- **Analysis Mode Selection**: Choose between "Whole Text Only" or "Detailed (Sentence-by-Sentence)" analysis

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

2. **GinZAモデルのインストール**:
```bash
pip install ja-ginza-electra
```

または、GinZAモデルに問題がある場合は：
```bash
fix_ginza.bat
```

3. **アプリケーションの起動**:
```bash
python app.py
```

4. **ブラウザでアクセス**:
   - Chromeブラウザで http://127.0.0.1:8000 にアクセスして使用できます
   - 推奨ブラウザ: Google Chrome（URL取得機能でX（Twitter）のリンクを使用する場合に必要）

### English

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Install GinZA model**:
```bash
pip install ja-ginza-electra
```

Or if you encounter issues with the GinZA model:
```bash
fix_ginza.bat
```

3. **Launch the application**:
```bash
python app.py
```

4. **Access in browser**:
   - Access http://127.0.0.1:8000 in Google Chrome browser to use the application
   - Recommended browser: Google Chrome (required when using URL fetching feature with X/Twitter links)

### Git LFSファイルのトラブルシューティング / Git LFS File Troubleshooting

**日本語 / Japanese**

GitHubからリポジトリをクローンまたはプルした後、アプリケーションを起動しようとするとエラーが発生する場合があります。これは、Git LFSで管理されている大きなファイル（モデルファイルなど）がポインタファイルのままになっているためです。

以下のコマンドを実行して、実際のファイルを取得してください：

```bash
git lfs install
git lfs fetch --all
git lfs checkout
```

**English**

After cloning or pulling the repository from GitHub, you may encounter errors when trying to launch the application. This is because large files managed by Git LFS (such as model files) remain as pointer files.

Run the following commands to retrieve the actual files:

```bash
git lfs install
git lfs fetch --all
git lfs checkout
```

---

## 使用方法 / How to Use

### 日本語 / Japanese

1. **著者グループの選択**
   - 「Author Group」から使用する著者グループを選択します
   - 「Former PMs (3)」: 元総理大臣3人のデータセット
   - 「NINJAL Corpus (16)」: NINJAL作文コーパスから抽出した16人のデータセット

2. **入力方法の選択**
   - **Direct Text Input**: テキストを直接入力
   - **URL Input**: URLを入力してテキストを自動取得（X（Twitter）のリンクも対応）

3. **テキストの入力**
   - 直接入力の場合: 「Input Text」欄に分析したい日本語の文章を入力します
   - URL入力の場合: URLを入力して「Fetch Text」ボタンをクリックします

4. **分析モードの選択**
   - **Whole Text Only**: 全体テキストのみを分析
   - **Detailed (Sentence-by-Sentence)**: 句読点ごとに細かく分析

5. **著者の選択**
   - 「Target Author」ドロップダウンから比較したい著者を選択します
   - 選択した著者グループに応じて、利用可能な著者リストが自動的に更新されます
   - 詳細分析モードでは、選択した著者との比較ではなく、全著者に対する分析が実行されます

6. **類似度の計算**
   - 「Calculate Similarity」ボタンをクリックします
   - または、テキスト入力欄で Ctrl+Enter（Macでは Cmd+Enter）を押すと実行できます

7. **結果の確認**
   - 右側の「Analysis Result」パネルに以下の情報が表示されます：
     - 選択した著者との類似度スコア（パーセンテージ）
     - すべての著者に対する予測確率（プログレスバー付き）
     - 分析結果（最大確率が50%未満の場合は「未知の著者の可能性」が表示されます）
   - 詳細分析モードの場合、各文ごとの分析結果も表示されます

### English

1. **Select Author Group**
   - Choose the author group from "Author Group"
   - "Former PMs (3)": Dataset of 3 former Japanese Prime Ministers
   - "NINJAL Corpus (16)": Dataset of 16 authors extracted from NINJAL Writing Corpus

2. **Select Input Method**
   - **Direct Text Input**: Enter text directly
   - **URL Input**: Enter URL to automatically fetch text (supports X/Twitter links)

3. **Enter Text**
   - For direct input: Input the Japanese text you want to analyze in the "Input Text" field
   - For URL input: Enter the URL and click the "Fetch Text" button

4. **Select Analysis Mode**
   - **Whole Text Only**: Analyze the entire text only
   - **Detailed (Sentence-by-Sentence)**: Analyze each sentence separated by punctuation

5. **Select Author**
   - Choose the author to compare from the "Target Author" dropdown
   - The available author list is automatically updated based on the selected author group
   - In detailed analysis mode, analysis is performed for all authors rather than comparing with the selected author

6. **Calculate Similarity**
   - Click the "Calculate Similarity" button
   - Or press Ctrl+Enter (Cmd+Enter on Mac) in the text input field

7. **View Results**
   - The "Analysis Result" panel on the right displays:
     - Similarity score with the selected author (percentage)
     - Prediction probabilities for all authors (with progress bars)
     - Analysis result (if the maximum probability is below 50%, a "Possible Unknown Author" warning is displayed)
   - In detailed analysis mode, analysis results for each sentence are also displayed

---

## ファイル構成 / File Structure

### 日本語 / Japanese

```
AA system_v5/
├── app.py                          # メインアプリケーション（FastAPI）
├── feature_extractor.py            # 特徴量抽出モジュール
├── model_loader.py                 # モデルローダー
├── requirements.txt                # 依存関係
├── README.md                       # このファイル
├── fix_ginza.bat                   # GinZAモデル再インストール用スクリプト
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
AA system_v5/
├── app.py                          # Main application (FastAPI)
├── feature_extractor.py            # Feature extraction module
├── model_loader.py                 # Model loader
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── fix_ginza.bat                   # Script for reinstalling GinZA model
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

6. **詳細分析（v5新機能）**
   - 句読点（「。」）でテキストを分割
   - 各文に対して全著者の類似度を計算
   - 各文ごとの最大確率著者を表示
   - 全体の分析結果も同時に表示

7. **ログ出力**
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

6. **Detailed Analysis (v5 New Feature)**
   - Splits text by punctuation marks ("。")
   - Calculates similarity for all authors for each sentence
   - Displays the author with maximum probability for each sentence
   - Also displays overall analysis results

7. **Logging**
   - Analysis results are recorded in `logs/similarity_log_YYYYMMDD.txt`

---

## APIエンドポイント / API Endpoints

### 日本語 / Japanese

このアプリケーションはRESTful APIも提供しています：

- `GET /`: メインページ（HTML）
- `GET /api/authors?author_group={group}`: 著者リストを取得
- `POST /api/predict`: 類似度を予測（全体テキスト分析）
  - リクエストボディ: `{"text": "...", "selected_author": "...", "author_group": "..."}`
  - レスポンス: JSON形式の予測結果
- `POST /api/fetch-url`: URLからテキストを取得（v5新機能）
  - リクエストボディ: `{"url": "..."}`
  - レスポンス: `{"text": "...", "success": true/false, "error": "..."}`
- `POST /api/predict-detailed`: 詳細分析を実行（v5新機能）
  - リクエストボディ: `{"text": "...", "author_group": "...", "analysis_mode": "whole"|"detailed"}`
  - レスポンス: JSON形式の詳細分析結果

### English

This application also provides RESTful API endpoints:

- `GET /`: Main page (HTML)
- `GET /api/authors?author_group={group}`: Get author list
- `POST /api/predict`: Predict similarity (whole text analysis)
  - Request body: `{"text": "...", "selected_author": "...", "author_group": "..."}`
  - Response: Prediction result in JSON format
- `POST /api/fetch-url`: Fetch text from URL (v5 new feature)
  - Request body: `{"url": "..."}`
  - Response: `{"text": "...", "success": true/false, "error": "..."}`
- `POST /api/predict-detailed`: Execute detailed analysis (v5 new feature)
  - Request body: `{"text": "...", "author_group": "...", "analysis_mode": "whole"|"detailed"}`
  - Response: Detailed analysis result in JSON format

---

## 注意事項 / Notes

### 日本語 / Japanese

- 初回起動時はモデルの読み込みに時間がかかる場合があります
- テキストは日本語で入力してください
- ログファイルは日付ごとに自動的に作成されます
- NINJAL Corpus (16)のモデルは、テキストパイプラインを使用するため、より多くの特徴量を処理します
- URL入力機能は、X（Twitter）のリンクにも対応しています（Seleniumを使用）
- X（Twitter）のリンクからテキストを取得する場合、初回実行時にChromeDriverが自動ダウンロードされます
- 詳細分析モードでは、各文ごとに分析を行うため、処理時間が長くなる場合があります

### English

- Model loading may take time on first launch
- Please input text in Japanese
- Log files are automatically created for each date
- The NINJAL Corpus (16) model processes more features as it uses text pipelines
- URL input feature supports X (Twitter) links (uses Selenium)
- When fetching text from X (Twitter) links, ChromeDriver will be automatically downloaded on first run
- Detailed analysis mode may take longer as it analyzes each sentence separately

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
- **requests, beautifulsoup4**: Web scraping (v5)
- **Selenium, webdriver-manager**: Browser automation for X/Twitter (v5)
