#!/usr/bin/env python3
"""Converted from return_uncorrelatedness.ipynb.

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


def analyze_return_autocorrelation(series, name, max_lag=40, alpha=0.05):
    """
    リターン系列に自己相関があるかどうかを判定する関数

    判定方法:
    1. 各ラグの自己相関係数を見る
    2. ACFプロットを見る
    3. Ljung-Box検定で、複数ラグをまとめて自己相関があるか確認する
    """

    r = series.replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) == 0:
        print(f"{name}: 有効なデータがありません")
        return None

    adjusted_max_lag = min(max_lag, max(len(r) - 1, 1))

    print("=" * 80)
    print(f"{name} の自己相関分析")
    print("=" * 80)

    print(f"データ数: {len(r)}")
    print(f"平均: {r.mean():.8f}")
    print(f"標準偏差: {r.std():.8f}")
    print()

    # ------------------------------------------------------------
    # 1. ラグごとの自己相関係数
    # ------------------------------------------------------------

    lags_to_check = [1, 2, 3, 5, 10, 20, 40]
    lags_to_check = [lag for lag in lags_to_check if lag <= adjusted_max_lag and lag < len(r)]

    acf_values = []

    print("ラグごとの自己相関係数")
    for lag in lags_to_check:
        acf = r.autocorr(lag=lag)
        acf_values.append({
            "lag": lag,
            "autocorrelation": acf,
            "abs_autocorrelation": abs(acf),
        })
        print(f"lag {lag:2d}: {acf:.6f}")

    acf_df = pd.DataFrame(acf_values)
    print()

    # ------------------------------------------------------------
    # 2. ACFプロット
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_acf(r, lags=adjusted_max_lag, ax=ax)
    ax.set_title(f"ACF of {name}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.grid(True)
    plt.show()

    # ------------------------------------------------------------
    # 3. Ljung-Box検定
    # ------------------------------------------------------------

    test_lags = [5, 10, 20, 40]
    test_lags = [lag for lag in test_lags if lag <= adjusted_max_lag and lag < len(r)]

    lb_result = acorr_ljungbox(r, lags=test_lags, return_df=True)

    print("Ljung-Box検定")
    print("帰無仮説 H0: 自己相関がない")
    print(lb_result)
    print()

    # ------------------------------------------------------------
    # 4. 簡易判定
    # ------------------------------------------------------------

    has_autocorr_by_ljungbox = (lb_result["lb_pvalue"] < alpha).any()
    has_large_individual_acf = (acf_df["abs_autocorrelation"] > 0.1).any()

    print("=" * 80)
    print("判定")
    print("=" * 80)

    if has_autocorr_by_ljungbox:
        print(f"Ljung-Box検定で p値 < {alpha} のラグがあります。")
        print("=> 統計的には、自己相関が存在する可能性があります。")
    else:
        print(f"Ljung-Box検定では p値 < {alpha} のラグはありません。")
        print("=> 強い自己相関は確認されませんでした。")

    print()

    if has_large_individual_acf:
        print("一部のラグで |自己相関| > 0.1 です。")
        print("=> 個別ラグではある程度の自己相関が見られます。")
    else:
        print("個別ラグの自己相関係数はおおむね小さいです。")
        print("=> リターンが自己相関を持たないという stylized fact と整合的です。")

    print()

    if (not has_autocorr_by_ljungbox) and (not has_large_individual_acf):
        final_judgement = "no_strong_autocorrelation"
        print("最終判定: 強い自己相関は確認されません。")
    elif has_autocorr_by_ljungbox and has_large_individual_acf:
        final_judgement = "autocorrelation_detected"
        print("最終判定: 自己相関がある可能性が高いです。")
    else:
        final_judgement = "weak_or_mixed_evidence"
        print("最終判定: 弱い自己相関、または判定が混在しています。")

    result = {
        "name": name,
        "n": len(r),
        "mean": r.mean(),
        "std": r.std(),
        "acf_table": acf_df,
        "ljungbox_result": lb_result,
        "has_autocorr_by_ljungbox": has_autocorr_by_ljungbox,
        "has_large_individual_acf": has_large_individual_acf,
        "final_judgement": final_judgement,
    }

    return result


def summarize_autocorrelation_result(result):
    """分析結果の辞書を、CSV保存しやすい1行のDataFrameに変換する。"""
    if result is None:
        return pd.DataFrame()

    row = {
        "name": result["name"],
        "n": result["n"],
        "mean": result["mean"],
        "std": result["std"],
        "has_autocorr_by_ljungbox": result["has_autocorr_by_ljungbox"],
        "has_large_individual_acf": result["has_large_individual_acf"],
        "final_judgement": result["final_judgement"],
    }

    for _, acf_row in result["acf_table"].iterrows():
        lag = int(acf_row["lag"])
        row[f"acf_lag{lag}"] = acf_row["autocorrelation"]
        row[f"abs_acf_lag{lag}"] = acf_row["abs_autocorrelation"]

    for lag, lb_row in result["ljungbox_result"].iterrows():
        lag = int(lag)
        row[f"lb_stat_lag{lag}"] = lb_row["lb_stat"]
        row[f"lb_pvalue_lag{lag}"] = lb_row["lb_pvalue"]

    return pd.DataFrame([row])


# ============================================================
# 2. 実データの読み込みと分析
# ============================================================

df = pd.read_csv(resolve_data_path("train_sp500_us10y.csv"))

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

# 数値変換
df["sp500"] = pd.to_numeric(df["sp500"], errors="coerce")
df["DGS10"] = pd.to_numeric(df["DGS10"], errors="coerce")

# すでに変化量系列なので、そのまま使う
sp500_return = df["sp500"].replace([np.inf, -np.inf], np.nan).dropna()
dgs10_change = df["DGS10"].replace([np.inf, -np.inf], np.nan).dropna()

sp500_acf_result = analyze_return_autocorrelation(
    sp500_return,
    name="S&P500 return",
    max_lag=40,
    alpha=0.05,
)

dgs10_acf_result = analyze_return_autocorrelation(
    dgs10_change,
    name="DGS10 yield change",
    max_lag=40,
    alpha=0.05,
)

real_uncorrelatedness_summary = pd.concat(
    [
        summarize_autocorrelation_result(sp500_acf_result),
        summarize_autocorrelation_result(dgs10_acf_result),
    ],
    ignore_index=True,
)

print("=" * 80)
print("実データ return uncorrelatedness summary")
print("=" * 80)
display(real_uncorrelatedness_summary)

# %% Cell 1
# ============================================================
# 3. 生成データ mixed_sabr / mixed_brown のリターン無相関性確認
#    - 標準出力を log.txt に保存
#    - 集計結果を CSV に保存
#    - 各系列のグラフを PNG に保存
# ============================================================

from contextlib import redirect_stdout
import re


def safe_filename(name):
    """ファイル名に使いにくい文字を置換する。"""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


def analyze_generated_return_uncorrelatedness(csv_path, dataset_name, fig_dir=None, max_lag=40, alpha=0.05):
    """
    mixedデータ内の mask*_sp500 / mask*_DGS10 を1系列ずつ自己相関分析する。

    - analyze_return_autocorrelation() の標準出力は呼び出し側でlogに保存される
    - analyze_return_autocorrelation() が作成したACFグラフは系列ごとにPNG保存する
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

            result = analyze_return_autocorrelation(
                generated_df[col],
                name=title,
                max_lag=max_lag,
                alpha=alpha,
            )
        finally:
            plt.show = original_show

        # analyze_return_autocorrelation() 内で作成されたACFグラフを保存
        if fig_dir is not None:
            new_fig_nums = [
                num for num in plt.get_fignums()
                if num not in before_fig_nums
            ]
            figure_labels = ["acf"]

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
            summary = summarize_autocorrelation_result(result)
            summary["dataset"] = dataset_name
            summary["column"] = col
            generated_results.append(summary)

    return pd.concat(generated_results, ignore_index=True) if generated_results else pd.DataFrame()


# ============================================================
# 出力先の設定
# ============================================================

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = output_dir / "figures_generated_return_uncorrelatedness"
fig_dir.mkdir(parents=True, exist_ok=True)

log_path = output_dir / "generated_return_uncorrelatedness_log.txt"
csv_output_path = output_dir / "generated_return_uncorrelatedness_summary.csv"


# ============================================================
# 分析実行
# ============================================================

with open(log_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):

        print("=" * 80)
        print("生成データ mixed_sabr / mixed_brown のリターン無相関性分析")
        print("=" * 80)

        mixed_sabr_path = resolve_data_path("mixed_sabr_masked.csv")
        mixed_brown_path = resolve_data_path("mixed_brown_masked.csv")

        print(f"mixed_sabr path : {mixed_sabr_path}")
        print(f"mixed_brown path: {mixed_brown_path}")
        print()

        mixed_sabr_uncorrelatedness_results = analyze_generated_return_uncorrelatedness(
            mixed_sabr_path,
            "mixed_sabr",
            fig_dir=fig_dir,
            max_lag=40,
            alpha=0.05,
        )

        mixed_brown_uncorrelatedness_results = analyze_generated_return_uncorrelatedness(
            mixed_brown_path,
            "mixed_brown",
            fig_dir=fig_dir,
            max_lag=40,
            alpha=0.05,
        )

        mixed_uncorrelatedness_results = pd.concat(
            [mixed_sabr_uncorrelatedness_results, mixed_brown_uncorrelatedness_results],
            ignore_index=True,
        )

        summary_columns = [
            "dataset",
            "column",
            "name",
            "n",
            "mean",
            "std",
            "acf_lag1",
            "acf_lag2",
            "acf_lag3",
            "acf_lag5",
            "acf_lag10",
            "acf_lag20",
            "acf_lag40",
            "abs_acf_lag1",
            "abs_acf_lag2",
            "abs_acf_lag3",
            "abs_acf_lag5",
            "abs_acf_lag10",
            "abs_acf_lag20",
            "abs_acf_lag40",
            "lb_stat_lag5",
            "lb_pvalue_lag5",
            "lb_stat_lag10",
            "lb_pvalue_lag10",
            "lb_stat_lag20",
            "lb_pvalue_lag20",
            "lb_stat_lag40",
            "lb_pvalue_lag40",
            "has_autocorr_by_ljungbox",
            "has_large_individual_acf",
            "final_judgement",
        ]

        summary_columns = [col for col in summary_columns if col in mixed_uncorrelatedness_results.columns]
        summary_df = mixed_uncorrelatedness_results[summary_columns]

        print()
        print("=" * 80)
        print("生成データ return uncorrelatedness 分析結果まとめ")
        print("=" * 80)
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
