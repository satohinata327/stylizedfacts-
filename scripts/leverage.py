#!/usr/bin/env python3
"""Converted from leverage.ipynb.

Stylized facts analysis script.
"""


# %% Cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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


def summarize_leverage_result(result_df, name):
    """lag別のレバレッジ結果を、系列ごとの特徴量1行にまとめる。"""
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame()

    corr = result_df["corr_return_future_squared_return"]
    row = {
        "name": name,
        "max_lag": int(result_df["lag"].max()),
        "avg_corr": corr.mean(),
        "min_corr": corr.min(),
        "max_corr": corr.max(),
        "negative_ratio": (corr < 0).mean(),
    }

    for _, lag_row in result_df.iterrows():
        lag = int(lag_row["lag"])
        row[f"corr_lag{lag}"] = lag_row["corr_return_future_squared_return"]

    if row["avg_corr"] < 0 and row["negative_ratio"] >= 0.7:
        row["final_judgement"] = "leverage_effect_detected"
    elif row["avg_corr"] < 0:
        row["final_judgement"] = "weak_leverage_effect"
    else:
        row["final_judgement"] = "no_clear_leverage_effect"

    return pd.DataFrame([row])


# ============================================================
# 2. レバレッジ効果を分析する関数
# ============================================================

def analyze_leverage_effect(r, name="S&P500 return", max_lag=20):
    """
    レバレッジ効果を確認する関数

    レバレッジ効果:
        現在のリターン r_t が負のとき、
        将来のボラティリティ r_{t+k}^2 が大きくなりやすい現象

    数学的には:
        Corr(r_t, r_{t+k}^2) < 0
    """

    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) == 0:
        print(f"{name}: 有効なデータがありません")
        return None

    adjusted_max_lag = min(max_lag, max(len(r) - 1, 1))
    results = []

    print("=" * 80)
    print(f"{name} のレバレッジ効果分析")
    print("=" * 80)

    print(f"データ数: {len(r)}")
    print(f"平均: {r.mean():.8f}")
    print(f"標準偏差: {r.std():.8f}")
    print()

    print("Corr(r_t, r_{t+k}^2)")
    print("-" * 80)

    for k in range(1, adjusted_max_lag + 1):
        future_vol = r.shift(-k) ** 2

        aligned = pd.concat([r, future_vol], axis=1).dropna()
        current_return = aligned.iloc[:, 0]
        future_squared_return = aligned.iloc[:, 1]

        corr = current_return.corr(future_squared_return)

        results.append({
            "lag": k,
            "corr_return_future_squared_return": corr,
        })

        print(f"lag {k:2d}: {corr:.6f}")

    result_df = pd.DataFrame(results)

    # ============================================================
    # 3. 簡易判定
    # ============================================================

    avg_corr = result_df["corr_return_future_squared_return"].mean()
    negative_ratio = (
        result_df["corr_return_future_squared_return"] < 0
    ).mean()

    print()
    print("=" * 80)
    print("判定")
    print("=" * 80)

    print(f"平均 Corr(r_t, r_(t+k)^2): {avg_corr:.6f}")
    print(f"負の相関になったラグの割合: {negative_ratio:.2%}")
    print()

    if avg_corr < 0 and negative_ratio >= 0.7:
        print("判定: レバレッジ効果が確認される可能性が高いです。")
        print("理由: 多くのラグで Corr(r_t, r_{t+k}^2) が負です。")
    elif avg_corr < 0:
        print("判定: 弱いレバレッジ効果が示唆されます。")
        print("理由: 平均的には負の相関ですが、ラグによって結果がばらついています。")
    else:
        print("判定: 明確なレバレッジ効果は確認しにくいです。")
        print("理由: Corr(r_t, r_{t+k}^2) が平均的に負ではありません。")

    # ============================================================
    # 4. 可視化
    # ============================================================

    plt.figure(figsize=(8, 4))
    plt.plot(
        result_df["lag"],
        result_df["corr_return_future_squared_return"],
        marker="o",
    )
    plt.axhline(0, linestyle="--")
    plt.title(f"{name}: Leverage Effect")
    plt.xlabel("Lag k")
    plt.ylabel("Corr(r_t, r_(t+k)^2)")
    plt.grid(True)
    plt.show()

    return result_df


# ============================================================
# 5. 実データの読み込みと分析
# ============================================================

df = pd.read_csv(resolve_data_path("train_sp500_us10y.csv"))

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

df["sp500"] = pd.to_numeric(df["sp500"], errors="coerce")

# レバレッジ効果は資産リターンと将来ボラティリティの関係なので、ここではsp500のみを対象にする
sp500_return = df["sp500"].replace([np.inf, -np.inf], np.nan).dropna()

leverage_result = analyze_leverage_effect(
    sp500_return,
    name="S&P500 return",
    max_lag=20,
)

real_leverage_summary = summarize_leverage_result(
    leverage_result,
    name="S&P500 return",
)

print("=" * 80)
print("実データ leverage effect summary")
print("=" * 80)
display(real_leverage_summary)

# %% Cell 1
# ============================================================
# 6. 生成データ mixed_sabr / mixed_brown のレバレッジ効果確認
#    - 標準出力を log.txt に保存
#    - lag別結果と集計結果を CSV に保存
#    - 各系列のグラフを PNG に保存
# ============================================================

from contextlib import redirect_stdout
import re


def safe_filename(name):
    """ファイル名に使いにくい文字を置換する。"""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


def analyze_generated_leverage_effect(csv_path, dataset_name, fig_dir=None, max_lag=20):
    """
    mixedデータ内の mask*_sp500 を1系列ずつレバレッジ効果分析する。

    レバレッジ効果は資産リターンと将来ボラティリティの負の関係なので、
    DGS10ではなくsp500系列だけを対象にする。
    """
    generated_df = pd.read_csv(csv_path)

    # 数値化できない値はNaNにして、分析関数側で除去する
    generated_df = generated_df.apply(pd.to_numeric, errors="coerce")

    target_columns = [
        col for col in generated_df.columns
        if col.endswith("_sp500")
    ]

    if fig_dir is not None:
        fig_dir = Path(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)

    lag_results = []
    summary_results = []

    for col in target_columns:
        title = f"{dataset_name}: {col}"
        before_fig_nums = set(plt.get_fignums())
        original_show = plt.show

        try:
            # 保存用の実行では show() でfigureが閉じられないようにする
            if fig_dir is not None:
                plt.show = lambda *args, **kwargs: None

            result = analyze_leverage_effect(
                generated_df[col],
                name=title,
                max_lag=max_lag,
            )
        finally:
            plt.show = original_show

        # analyze_leverage_effect() 内で作成されたグラフを保存
        if fig_dir is not None:
            new_fig_nums = [
                num for num in plt.get_fignums()
                if num not in before_fig_nums
            ]
            figure_labels = ["leverage_correlation"]

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
            result["name"] = title
            lag_results.append(result)

            summary = summarize_leverage_result(result, name=title)
            summary["dataset"] = dataset_name
            summary["column"] = col
            summary_results.append(summary)

    lag_df = pd.concat(lag_results, ignore_index=True) if lag_results else pd.DataFrame()
    summary_df = pd.concat(summary_results, ignore_index=True) if summary_results else pd.DataFrame()

    return lag_df, summary_df


# ============================================================
# 出力先の設定
# ============================================================

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = output_dir / "figures_generated_leverage"
fig_dir.mkdir(parents=True, exist_ok=True)

log_path = output_dir / "generated_leverage_log.txt"
lag_csv_output_path = output_dir / "generated_leverage_lag_results.csv"
summary_csv_output_path = output_dir / "generated_leverage_summary.csv"


# ============================================================
# 分析実行
# ============================================================

with open(log_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):

        print("=" * 80)
        print("生成データ mixed_sabr / mixed_brown のレバレッジ効果分析")
        print("=" * 80)
        print("対象系列: mask*_sp500 のみ")
        print()

        mixed_sabr_path = resolve_data_path("mixed_sabr_masked.csv")
        mixed_brown_path = resolve_data_path("mixed_brown_masked.csv")

        print(f"mixed_sabr path : {mixed_sabr_path}")
        print(f"mixed_brown path: {mixed_brown_path}")
        print()

        mixed_sabr_leverage_lag_results, mixed_sabr_leverage_summary = analyze_generated_leverage_effect(
            mixed_sabr_path,
            "mixed_sabr",
            fig_dir=fig_dir,
            max_lag=20,
        )

        mixed_brown_leverage_lag_results, mixed_brown_leverage_summary = analyze_generated_leverage_effect(
            mixed_brown_path,
            "mixed_brown",
            fig_dir=fig_dir,
            max_lag=20,
        )

        mixed_leverage_lag_results = pd.concat(
            [mixed_sabr_leverage_lag_results, mixed_brown_leverage_lag_results],
            ignore_index=True,
        )

        mixed_leverage_summary = pd.concat(
            [mixed_sabr_leverage_summary, mixed_brown_leverage_summary],
            ignore_index=True,
        )

        summary_columns = [
            "dataset",
            "column",
            "name",
            "max_lag",
            "avg_corr",
            "min_corr",
            "max_corr",
            "negative_ratio",
            "final_judgement",
        ]
        summary_columns = [col for col in summary_columns if col in mixed_leverage_summary.columns]
        summary_df = mixed_leverage_summary[summary_columns]

        print()
        print("=" * 80)
        print("生成データ leverage effect 分析結果まとめ")
        print("=" * 80)
        print(summary_df.to_string(index=False))


# ============================================================
# CSV保存
# ============================================================

mixed_leverage_lag_results.to_csv(
    lag_csv_output_path,
    index=False,
    encoding="utf-8-sig",
)

mixed_leverage_summary.to_csv(
    summary_csv_output_path,
    index=False,
    encoding="utf-8-sig",
)


# ============================================================
# Notebook上にも結果を表示
# ============================================================

print(f"ログを保存しました: {log_path}")
print(f"lag別CSVを保存しました: {lag_csv_output_path}")
print(f"集計CSVを保存しました: {summary_csv_output_path}")
print(f"グラフを保存しました: {fig_dir}")

display(summary_df)
