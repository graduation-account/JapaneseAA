# Gitセットアップ手順

GitとGit LFSを使用してGitHubにアップロードする手順です。

## 前提条件

- Git for Windowsがインストール済み（完了済み）
- GitHubリポジトリが作成済み（完了済み: https://github.com/graduation-account/JapaneseAA）

## 手順

### 方法1: バッチファイルを使用（推奨）

1. `setup_git.bat` をダブルクリックして実行
2. 管理者権限のプロンプトが表示された場合は「はい」を選択
3. 完了まで待機

### 方法2: 手動でコマンドを実行

新しいPowerShellまたはコマンドプロンプトを開き、以下のコマンドを順番に実行してください：

```powershell
# プロジェクトディレクトリに移動
cd "D:\AA system"

# Git LFSのインストール
git lfs install

# Gitリポジトリの初期化
git init

# .gitattributesを追加（Git LFS設定）
git add .gitattributes
git commit -m "Add Git LFS configuration"

# すべてのファイルを追加
git add .

# 初回コミット
git commit -m "Initial commit: Japanese Author Similarity Analysis System"

# リモートリポジトリの追加
git remote add origin https://github.com/graduation-account/JapaneseAA.git

# メインブランチに設定
git branch -M main

# プッシュ（Personal Access Tokenを使用）
# ユーザー名: graduation-account
# パスワード: [Personal Access Tokenを入力してください]
git push -u origin main
```

## 注意事項

- モデルファイルは大きいため、アップロードに時間がかかる場合があります
- Personal Access Tokenは機密情報です。GitHubにプッシュした後は、トークンを無効化することを推奨します
- 初回プッシュ時、GitHubの認証情報を求められた場合は、上記のPersonal Access Tokenを入力してください

