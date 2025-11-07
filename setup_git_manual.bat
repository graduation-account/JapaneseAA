@echo off
echo ========================================
echo Git Setup for JapaneseAA Repository
echo ========================================
echo.

REM 現在のディレクトリに移動
cd /d "%~dp0"

REM Gitのインストール場所を確認
set "GIT_PATH="
if exist "C:\Program Files\Git\bin\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\bin"
) else if exist "C:\Program Files\Git\cmd\git.exe" (
    set "GIT_PATH=C:\Program Files\Git\cmd"
) else (
    echo ERROR: Git is not installed or not found in standard locations.
    echo Please install Git for Windows from https://git-scm.com/download/win
    pause
    exit /b 1
)

echo Found Git at: %GIT_PATH%
echo.

REM PATHに追加
set "PATH=%PATH%;%GIT_PATH%"

REM Git LFSのインストール
echo Installing Git LFS...
"%GIT_PATH%\git.exe" lfs install
if errorlevel 1 (
    echo ERROR: Failed to install Git LFS
    pause
    exit /b 1
)

REM Gitリポジトリの初期化
echo Initializing Git repository...
"%GIT_PATH%\git.exe" init
if errorlevel 1 (
    echo ERROR: Failed to initialize Git repository
    pause
    exit /b 1
)

REM .gitattributesを追加
echo Adding .gitattributes...
"%GIT_PATH%\git.exe" add .gitattributes
"%GIT_PATH%\git.exe" commit -m "Add Git LFS configuration"
if errorlevel 1 (
    echo WARNING: Failed to commit .gitattributes (may already be committed)
)

REM すべてのファイルを追加
echo Adding all files...
"%GIT_PATH%\git.exe" add .
if errorlevel 1 (
    echo ERROR: Failed to add files
    pause
    exit /b 1
)

REM 初回コミット
echo Creating initial commit...
"%GIT_PATH%\git.exe" commit -m "Initial commit: Japanese Author Similarity Analysis System"
if errorlevel 1 (
    echo ERROR: Failed to create commit
    pause
    exit /b 1
)

REM リモートリポジトリの追加（既に存在する場合はスキップ）
echo Adding remote repository...
"%GIT_PATH%\git.exe" remote remove origin 2>nul
"%GIT_PATH%\git.exe" remote add origin https://github.com/graduation-account/JapaneseAA.git
if errorlevel 1 (
    echo ERROR: Failed to add remote repository
    pause
    exit /b 1
)

REM メインブランチに設定
echo Setting main branch...
"%GIT_PATH%\git.exe" branch -M main
if errorlevel 1 (
    echo WARNING: Failed to rename branch (may already be main)
)

REM プッシュ
echo.
echo ========================================
echo Ready to push to GitHub
echo ========================================
echo.
echo When prompted for credentials:
echo   Username: graduation-account
echo   Password: [Personal Access Tokenを入力してください]
echo.
echo Pushing to GitHub (this may take a while due to large model files)...
"%GIT_PATH%\git.exe" push -u origin main
if errorlevel 1 (
    echo.
    echo ERROR: Failed to push to GitHub
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS! Repository uploaded to GitHub
echo ========================================
echo.
echo Repository URL: https://github.com/graduation-account/JapaneseAA
echo.
pause

