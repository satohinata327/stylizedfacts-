# Stylized Facts Analysis

S&P500日次リターンと米10年国債利回り差分を対象に、金融時系列の stylized facts を確認するための分析リポジトリです。

もともとは notebook で作成していた分析を、GitHub上で読みやすく、再実行しやすいように Python スクリプトへ整理しています。

## 1. 分析内容

このリポジトリでは、以下の stylized facts を確認します。

```text
fattail                  fat tail / 極端値の出やすさ
return_uncorrelatedness  リターン系列の自己相関の弱さ
volatilityclustering     絶対値・二乗リターンの自己相関
leverage                 リターンと将来ボラティリティの関係
asymmetryofreturns       正負リターンや上下tailの非対称性
```

## 2. ディレクトリ構成

```text
stylizedfacts/
  scripts/
    fattail.py
    return_uncorrelatedness.py
    volatilityclustering.py
    leverage.py
    asymmetryofreturns.py
  data/
    train_sp500_us10y.csv
    mixed_brown_masked.csv
    mixed_sabr_masked.csv
  results/
  requirements.txt
  README.md
```

`data/` と `results/` はGit管理しない想定です。手元で実行する際に必要なCSVを `data/` に配置してください。

## 3. 入力データ

以下のCSVを `data/` に置いてください。

```text
data/train_sp500_us10y.csv
data/mixed_brown_masked.csv
data/mixed_sabr_masked.csv
```

`train_sp500_us10y.csv` は以下の列を持つ想定です。

```text
date,sp500,DGS10
```

`mixed_brown_masked.csv` と `mixed_sabr_masked.csv` は以下のような列を持つ想定です。

```text
mask1_sp500,mask1_DGS10,mask2_sp500,mask2_DGS10,...
```

## 4. セットアップ

Python仮想環境を使う場合の例です。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. 実行方法

リポジトリ直下で、各スクリプトを実行します。

```bash
python scripts/fattail.py
python scripts/return_uncorrelatedness.py
python scripts/volatilityclustering.py
python scripts/leverage.py
python scripts/asymmetryofreturns.py
```

各スクリプトは、`data/` からCSVを読み込み、`results/` にログ、集計CSV、グラフを出力します。

## 6. 出力例

代表的な出力は以下です。

```text
results/generated_fattail_log.txt
results/generated_fattail_summary.csv
results/figures_generated_fattail/

results/generated_return_uncorrelatedness_log.txt
results/generated_return_uncorrelatedness_summary.csv
results/figures_generated_return_uncorrelatedness/

results/generated_volatility_clustering_log.txt
results/generated_volatility_clustering_summary.csv
results/figures_generated_volatility_clustering/

results/generated_leverage_log.txt
results/generated_leverage_lag_results.csv
results/generated_leverage_summary.csv
results/figures_generated_leverage/

results/generated_asymmetry_log.txt
results/generated_asymmetry_summary.csv
results/figures_generated_asymmetry/
```

## 7. Notebookについて

元の `.ipynb` は GitHub 上では差分や内容を追いにくいため、同等の処理を `scripts/*.py` に変換しています。

今後の編集は、基本的に `scripts/` 配下の Python ファイルに対して行う想定です。

## 8. 注意

このリポジトリは stylized facts の確認・可視化を目的とした分析コードです。

Mahalanobis距離やHotelling's T²による判別実験は、別リポジトリ `mahalanobis` 側で扱います。
