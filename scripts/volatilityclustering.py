#!/usr/bin/env python3
"""Converted from volatilityclustering.ipynb.

Stylized facts analysis script.
"""


# %% Cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# 1. 共通関数
# ============================================================

def resolve_data_path(filename):
    """Notebookの実行場所に依存しすぎないようにCSVの場所を探す。"""
    candidates = [
        Path(filename),
        Path("data") / filename,
        Path("stylizedfacts-") / "data" / filename,
        Path("mixed_data_1st") / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} が見つかりません: {candidates}")


def print_autocorrelation(series, name, lags=(1, 5, 10, 20, 40)):
    print("=" * 70)
    print(f"{name} の自己相関")
    print("=" * 70)

    for lag in lags:
        if lag < len(series):
            acf_value = series.autocorr(lag=lag)
            print(f"lag {lag:2d}: {acf_value:.6f}")
        else:
            print(f"lag {lag:2d}: データ数不足")

    print()


def analyze_volatility_clustering(series, name, max_lag=40, lags=(1, 5, 10, 20, 40)):
    """
    1つの時系列についてボラティリティ・クラスタリングを分析する。

    確認するもの:
    - 元系列の時系列プロット
    - 絶対値系列 |x_t| の時系列プロット
    - 二乗系列 x_t^2 の時系列プロット
    - 元系列 / 絶対値系列 / 二乗系列のACF
    - 指定lagでの自己相関
    - Ljung-Box検定
    """
    x = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        print(f"{name}: 有効なデータがありません")
        return None

    abs_x = x.abs()
    squared_x = x ** 2
    adjusted_max_lag = min(max_lag, max(len(x) - 1, 1))
    lags = tuple(lag for lag in lags if lag < len(x))

    print("=" * 70)
    print(f"{name} のボラティリティ・クラスタリング分析")
    print("=" * 70)
    print(f"データ数: {len(x)}")
    print(f"平均: {x.mean():.8f}")
    print(f"標準偏差: {x.std():.8f}")
    print()

    # ------------------------------------------------------------
    # 時系列プロット
    # ------------------------------------------------------------
    plt.figure(figsize=(10, 4))
    plt.plot(x)
    plt.title(f"{name}: Series")
    plt.xlabel("Time")
    plt.ylabel(name)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(abs_x)
    plt.title(f"{name}: Absolute Series |x_t|")
    plt.xlabel("Time")
    plt.ylabel("|x_t|")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(squared_x)
    plt.title(f"{name}: Squared Series x_t^2")
    plt.xlabel("Time")
    plt.ylabel("x_t^2")
    plt.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # ACFプロット
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(x, lags=adjusted_max_lag, ax=ax)
    ax.set_title(f"{name}: ACF of Series")
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(abs_x, lags=adjusted_max_lag, ax=ax)
    ax.set_title(f"{name}: ACF of Absolute Series")
    ax.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(squared_x, lags=adjusted_max_lag, ax=ax)
    ax.set_title(f"{name}: ACF of Squared Series")
    ax.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # 数値で自己相関を確認
    # ------------------------------------------------------------
    print_autocorrelation(x, f"{name}: series x_t", lags=lags)
    print_autocorrelation(abs_x, f"{name}: absolute series |x_t|", lags=lags)
    print_autocorrelation(squared_x, f"{name}: squared series x_t^2", lags=lags)

    # ------------------------------------------------------------
    # Ljung-Box検定
    # ------------------------------------------------------------
    lags_to_test = [lag for lag in (5, 10, 20, 40) if lag < len(x)]

    lb_series = acorr_ljungbox(x, lags=lags_to_test, return_df=True)
    lb_abs = acorr_ljungbox(abs_x, lags=lags_to_test, return_df=True)
    lb_squared = acorr_ljungbox(squared_x, lags=lags_to_test, return_df=True)

    print("=" * 70)
    print(f"Ljung-Box test for {name}: series")
    print("=" * 70)
    print(lb_series)
    print()

    print("=" * 70)
    print(f"Ljung-Box test for {name}: absolute series")
    print("=" * 70)
    print(lb_abs)
    print()

    print("=" * 70)
    print(f"Ljung-Box test for {name}: squared series")
    print("=" * 70)
    print(lb_squared)
    print()

    # ------------------------------------------------------------
    # 判定用まとめ
    # ------------------------------------------------------------
    transforms = {
        "series": x,
        "absolute_series": abs_x,
        "squared_series": squared_x,
    }
    lb_tables = {
        "series": lb_series,
        "absolute_series": lb_abs,
        "squared_series": lb_squared,
    }

    rows = []
    for transform_name, transformed in transforms.items():
        row = {
            "name": name,
            "transform": transform_name,
            "n": len(transformed),
            "mean": transformed.mean(),
            "std": transformed.std(),
        }

        for lag in (1, 5, 10, 20, 40):
            row[f"acf_lag{lag}"] = transformed.autocorr(lag=lag) if lag < len(transformed) else np.nan

        lb_table = lb_tables[transform_name]
        for lag in lags_to_test:
            row[f"lb_stat_lag{lag}"] = lb_table.loc[lag, "lb_stat"]
            row[f"lb_pvalue_lag{lag}"] = lb_table.loc[lag, "lb_pvalue"]

        rows.append(row)

    summary = pd.DataFrame(rows)

    print("=" * 70)
    print(f"{name} volatility clustering summary")
    print("=" * 70)
    print(summary)

    abs_acf_lag1 = abs_x.autocorr(lag=1) if len(abs_x) > 1 else np.nan
    sq_acf_lag1 = squared_x.autocorr(lag=1) if len(squared_x) > 1 else np.nan

    print()
    print("=" * 70)
    print("簡易判定")
    print("=" * 70)

    if abs_acf_lag1 > 0.1 or sq_acf_lag1 > 0.1:
        print("絶対値系列または二乗系列に正の自己相関が見られます。")
        print("ボラティリティ・クラスタリングが存在する可能性があります。")
    else:
        print("lag 1 だけを見ると、強いボラティリティ・クラスタリングは確認しにくいです。")

    return summary


# ============================================================
# 2. 実データの読み込みと分析
# ============================================================

df = pd.read_csv(resolve_data_path("train_sp500_us10y.csv"))

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

df["sp500"] = pd.to_numeric(df["sp500"], errors="coerce")
df["DGS10"] = pd.to_numeric(df["DGS10"], errors="coerce")

real_sp500_summary = analyze_volatility_clustering(
    df["sp500"],
    "S&P500 return",
)

real_dgs10_summary = analyze_volatility_clustering(
    df["DGS10"],
    "DGS10 yield change",
)

real_volatility_summary = pd.concat(
    [real_sp500_summary, real_dgs10_summary],
    ignore_index=True,
)

print("=" * 70)
print("実データ volatility clustering summary")
print("=" * 70)
display(real_volatility_summary)

# %% Cell 1
# ============================================================
# 3. 生成データ mixed_sabr / mixed_brown のボラティリティ・クラスタリング確認
#    - 標準出力を log.txt に保存
#    - 集計結果を CSV に保存
#    - 各系列のグラフを PNG に保存
# ============================================================

from contextlib import redirect_stdout
import re


def safe_filename(name):
    """ファイル名に使いにくい文字を置換する。"""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


def analyze_generated_volatility_clustering(csv_path, dataset_name, fig_dir=None):
    """
    mixedデータ内の mask*_sp500 / mask*_DGS10 を1系列ずつ分析する。

    - analyze_volatility_clustering() の標準出力は呼び出し側でlogに保存される
    - analyze_volatility_clustering() が作成したグラフは系列ごとにPNG保存する
    - 返り値は各系列の統計量をまとめたDataFrame
    """
    generated_df = pd.read_csv(csv_path)

    # 数値化できない値はNaNにして、分析関数側で除去する
    generated_df = generated_df.apply(pd.to_numeric, errors="coerce")

    target_columns = [
        col for col in generated_df.columns
        if col.endswith("_sp500") or col.endswith("_DGS10")
    ]

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

    generated_results = []

    for col in target_columns:
        title = f"{dataset_name}: {col}"
        before_fig_nums = set(plt.get_fignums())
        original_show = plt.show

        try:
            # 保存用の実行では show() でfigureが閉じられないようにする
            if fig_dir is not None:
                plt.show = lambda *args, **kwargs: None

            result = analyze_volatility_clustering(
                generated_df[col],
                title,
            )
        finally:
            plt.show = original_show

        # analyze_volatility_clustering() 内で作成されたグラフをすべて保存
        if fig_dir is not None:
            new_fig_nums = [
                num for num in plt.get_fignums()
                if num not in before_fig_nums
            ]
            figure_labels = [
                "series",
                "absolute_series",
                "squared_series",
                "acf_series",
                "acf_absolute_series",
                "acf_squared_series",
            ]

            if not new_fig_nums:
                print(f"Warning: no figure was created for {title}")

            for idx, fig_num in enumerate(new_fig_nums, start=1):
                fig = plt.figure(fig_num)
                label = figure_labels[idx - 1] if idx <= len(figure_labels) else f"figure_{idx}"
                fig_path = fig_dir / (
                    f"{safe_filename(dataset_name)}_{safe_filename(col)}_{label}.png"
                )
                fig.savefig(fig_path, dpi=300, bbox_inches="tight")
                plt.close(fig)

                print(f"Saved figure: {fig_path}")

        if result is not None:
            result = result.copy()
            result["dataset"] = dataset_name
            result["column"] = col
            generated_results.append(result)

    return pd.concat(generated_results, ignore_index=True) if generated_results else pd.DataFrame()


# ============================================================
# 出力先の設定
# ============================================================

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = output_dir / "figures_generated_volatility_clustering"
fig_dir.mkdir(parents=True, exist_ok=True)

log_path = output_dir / "generated_volatility_clustering_log.txt"
csv_output_path = output_dir / "generated_volatility_clustering_summary.csv"


# ============================================================
# 分析実行
# ============================================================

with open(log_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):

        print("=" * 70)
        print("生成データ mixed_sabr / mixed_brown のボラティリティ・クラスタリング分析")
        print("=" * 70)

        mixed_sabr_path = resolve_data_path("mixed_sabr_masked.csv")
        mixed_brown_path = resolve_data_path("mixed_brown_masked.csv")

        print(f"mixed_sabr path : {mixed_sabr_path}")
        print(f"mixed_brown path: {mixed_brown_path}")
        print()

        mixed_sabr_volatility_results = analyze_generated_volatility_clustering(
            mixed_sabr_path,
            "mixed_sabr",
            fig_dir=fig_dir,
        )

        mixed_brown_volatility_results = analyze_generated_volatility_clustering(
            mixed_brown_path,
            "mixed_brown",
            fig_dir=fig_dir,
        )

        mixed_volatility_results = pd.concat(
            [mixed_sabr_volatility_results, mixed_brown_volatility_results],
            ignore_index=True,
        )

        summary_columns = [
            "dataset",
            "column",
            "name",
            "transform",
            "n",
            "mean",
            "std",
            "acf_lag1",
            "acf_lag5",
            "acf_lag10",
            "acf_lag20",
            "acf_lag40",
            "lb_stat_lag5",
            "lb_pvalue_lag5",
            "lb_stat_lag10",
            "lb_pvalue_lag10",
            "lb_stat_lag20",
            "lb_pvalue_lag20",
            "lb_stat_lag40",
            "lb_pvalue_lag40",
        ]

        summary_columns = [col for col in summary_columns if col in mixed_volatility_results.columns]
        summary_df = mixed_volatility_results[summary_columns]

        print()
        print("=" * 70)
        print("生成データ volatility clustering 分析結果まとめ")
        print("=" * 70)
        print(summary_df.to_string(index=False))


# ============================================================
# CSV保存
# ============================================================

summary_df.to_csv(
    csv_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Notebook上にも結果を表示
# ============================================================

print(f"ログを保存しました: {log_path}")
print(f"CSVを保存しました: {csv_output_path}")
print(f"グラフを保存しました: {fig_dir}")

display(summary_df)
