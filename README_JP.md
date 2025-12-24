# Ternary Plot Visualizer

学術論文・プレゼンテーション向けの三成分組成図可視化ツール

## 機能

- **インタラクティブな三角図**: Plotlyを使用した三成分組成データの可視化
- **データ読み込み**: CSV、TXT、TSVファイルに対応（区切り文字自動検出）
- **編集可能なデータテーブル**: ブラウザ上でデータポイントの追加・編集・削除が可能
- **組成式サポート**:
  - pymatgenを使用した自動還元式計算（A+B+C → 還元式）
  - 組成式分解機能（Li3PS4のような式を入力するとA/B/C係数を自動計算）
  - 手動入力モード（A、B、C値を直接入力）
- **論文品質の図作成**:
  - 図サイズ、フォント、色のカスタマイズ
  - 複数のカラースケール（Turbo、Viridis、Jet等）、離散/連続オプション
  - 化学式の自動下付き文字変換（Li2O → Li₂O）
  - カラーバーのカスタマイズ（サイズ、位置、目盛り、タイトル配置）
  - 高解像度PNG出力（4倍スケール）
- **補間機能**: データポイント間の滑らかなヒートマップ補間
- **Z値ヒートマップ**: プリセットラベル（σ、Ea、容量）またはカスタムラベルでの色分け表示

## インストール

```bash
# リポジトリをクローン
git clone https://github.com/matsui-naoki/ternary-app.git
cd ternary-app

# 依存関係をインストール
pip install -r requirements.txt
```

## 使い方

```bash
streamlit run app.py
```

ブラウザで http://localhost:8501 を開いてください。

## データ形式

CSV/TXTファイルを読み込み可能：
- **列**: A、B、C（組成比）、オプションでZ（物性値）、Name（サンプル名）
- **ヘッダー**: 最初の行がテキストの場合、列ヘッダーとして扱われます
- **区切り文字**: 自動検出（カンマ、タブ、スペース）

例：
```csv
Li2S,P2S5,LiI,sigma,Name
0.75,0.25,0.0,0.5,Sample1
0.70,0.20,0.10,1.2,Sample2
0.60,0.30,0.10,2.5,Sample3
```

## 主な操作

### Data Loader
- **Browse files**: CSVファイルをアップロード
- **Template**: 空のテンプレートCSVをダウンロード
- **Load Sample**: サンプルデータを読み込み

### Data Labels
- A、B、C、Zのラベルをカスタマイズ
- Zプリセット: σ（導電率）、Ea（活性化エネルギー）、容量

### Add Data
- **Formula input ON**: 化学式を入力してA、B、Cに自動分解
- **Formula input OFF**: A、B、C値を直接入力（合計1に正規化）

### Plot Settings
- **General**: 図サイズ、フォント、軸設定
- **Markers**: マーカーサイズ、形状、透明度
- **Colorbar**: カラースケール、範囲、離散化
- **Heatmap**: 補間解像度、マーカーシンボル

## 出力

- **PNG**: ツールバーのカメラアイコンをクリック（4倍解像度）
- **HTML**: インタラクティブなHTMLファイル
- **CSV**: データテーブルのエクスポート

## 必要要件

- Python 3.8以上
- streamlit
- numpy
- pandas
- plotly
- scipy
- pymatgen
- sympy
- kaleido（PNG出力用、オプション）

## ライセンス

MIT License
