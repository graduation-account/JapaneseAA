@echo off
cd /d "%~dp0"

REM Gitのパスを追加
set "PATH=%PATH%;C:\Program Files\Git\bin;C:\Program Files\Git\cmd"

REM Git LFSのインストール
git lfs install

REM Gitリポジトリの初期化
git init

REM .gitattributesを追加
git add .gitattributes
git commit -m "Add Git LFS configuration"

REM すべてのファイルを追加
git add .

REM 初回コミット
git commit -m "Initial commit: Japanese Author Similarity Analysis System"

REM リモートリポジトリの追加
git remote add origin https://github.com/graduation-account/JapaneseAA.git

REM メインブランチに設定
git branch -M main

REM プッシュ
git push -u origin main

pause

