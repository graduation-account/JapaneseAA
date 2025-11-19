@echo off
REM Git LFSで管理されているファイルを実際のファイルに置き換えるスクリプト
echo Git LFSファイルをチェックアウトしています...
git lfs install
git lfs fetch --all
git lfs checkout
echo 完了しました。

