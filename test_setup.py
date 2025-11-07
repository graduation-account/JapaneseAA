"""環境と依存関係のテストスクリプト"""
import sys

def test_imports():
    """必要なライブラリがインストールされているか確認"""
    print("=" * 50)
    print("依存関係のテスト")
    print("=" * 50)
    
    modules = [
        ("gradio", "gradio"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("catboost", "catboost"),
        ("sklearn", "sklearn"),
        ("joblib", "joblib"),
        ("spacy", "spacy"),
    ]
    
    failed = []
    for name, module in modules:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - インストールされていません")
            failed.append(name)
    
    if failed:
        print(f"\n以下のパッケージをインストールしてください:")
        print(f"pip install {' '.join(failed)}")
        return False
    
    print("\n✓ すべての依存関係がインストールされています")
    return True

def test_ginza():
    """GinZAモデルが利用可能か確認"""
    print("\n" + "=" * 50)
    print("GinZAモデルのテスト")
    print("=" * 50)
    
    try:
        import spacy
        nlp = spacy.load("ja_ginza_electra")
        print("✓ GinZAモデルが利用可能です")
        return True
    except OSError:
        print("✗ GinZAモデルが見つかりません")
        print("\n以下のコマンドでインストールしてください:")
        print("python -m spacy download ja_ginza_electra")
        return False
    except Exception as e:
        print(f"✗ GinZAの読み込みエラー: {e}")
        return False

def test_files():
    """必要なファイルが存在するか確認"""
    print("\n" + "=" * 50)
    print("ファイルの存在確認")
    print("=" * 50)
    
    from pathlib import Path
    base_dir = Path(__file__).parent
    
    files = [
        ("設定ファイル", base_dir / "models" / "ensemble_config.json"),
        ("データセット", base_dir / "dataset" / "minister_features.csv"),
        ("BERTモデル", base_dir / "models" / "bert_minister"),
        ("CatBoostモデル", base_dir / "models" / "catboost_minister.cbm"),
        ("Label Encoder", base_dir / "models" / "preprocess" / "label_encoder.joblib"),
        ("Chi Selector", base_dir / "models" / "preprocess" / "chi_selector.joblib"),
        ("feature_extractor.py", base_dir / "feature_extractor.py"),
        ("model_loader.py", base_dir / "model_loader.py"),
    ]
    
    failed = []
    for name, path in files:
        if path.exists():
            print(f"✓ {name}: {path}")
        else:
            print(f"✗ {name}: {path} - 見つかりません")
            failed.append(name)
    
    if failed:
        print(f"\n以下のファイルが見つかりませんでした:")
        for name in failed:
            print(f"  - {name}")
        return False
    
    print("\n✓ すべての必要なファイルが存在します")
    return True

if __name__ == "__main__":
    results = []
    results.append(test_imports())
    results.append(test_ginza())
    results.append(test_files())
    
    print("\n" + "=" * 50)
    if all(results):
        print("✓ すべてのテストが成功しました！")
        print("python app.py を実行できます。")
        sys.exit(0)
    else:
        print("✗ 一部のテストが失敗しました。")
        print("上記のエラーを修正してから再度実行してください。")
        sys.exit(1)


