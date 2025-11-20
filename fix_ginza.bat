@echo off
echo ========================================
echo GinZAモデルの再インストールスクリプト
echo ========================================
echo.

echo [1/3] 既存のGinZAモデルをアンインストール中...
pip uninstall -y ja-ginza-electra
echo.

echo [2/3] パッケージを再インストール中...
pip install -U ginza transformers spacy
echo.

echo [3/3] GinZAモデルをダウンロード中...
python -m spacy download ja_ginza_electra
echo.

echo ========================================
echo インストール完了！
echo ========================================
pause

