#!/usr/bin/env python3
"""Converted from asymmetryofreturns.ipynb.

Stylized facts analysis script.
"""


# %% Cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. データ読み込み
# ============================================================

def resolve_data_path(filename):
    candidates = [
        Path(filename),
        Path("data") / filename,
        Path("stylizedfacts-") / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"{filename} が見つかりません: {candidates}")


df = pd.read_csv(resolve_data_path("train_sp500_us10y.csv"))

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.set_index("date")

# 数値変換
df["sp500"] = pd.to_numeric(df["sp500"], errors="coerce")

# sp500 はすでにリターン系列
r = df["sp500"].replace([np.inf, -np.inf], np.nan).dropna()


# ============================================================
# 2. リターン非対称性の分析関数
# ============================================================

def analyze_return_asymmetry(r, name="Return"):
    r = r.replace([np.inf, -np.inf], np.nan).dropna()

    if len(r) == 0:
        print("有効なデータがありません")
        return None

    # -----------------------------
    # 基本統計量
    # -----------------------------
    mean = r.mean()
    std = r.std()
    skewness = r.skew()

    # 正のリターン・負のリターンに分ける
    positive_returns = r[r > 0]
    negative_returns = r[r < 0]

    # 上昇時の平均値幅
    avg_up = positive_returns.mean()

    # 下落時の平均値幅
    # 負の値の絶対値を取る
    avg_down = negative_returns.abs().mean()

    # 上昇時・下落時の中央値値幅
    median_up = positive_returns.median()
    median_down = negative_returns.abs().median()

    # 最大上昇・最大下落
    max_up = r.max()
    max_down = abs(r.min())

    # 下落値幅 / 上昇値幅
    avg_down_up_ratio = avg_down / avg_up if avg_up != 0 else np.nan
    median_down_up_ratio = median_down / median_up if median_up != 0 else np.nan
    max_down_up_ratio = max_down / max_up if max_up != 0 else np.nan

    # -----------------------------
    # 分位点ベースの非対称性
    # -----------------------------
    q01 = r.quantile(0.01)
    q05 = r.quantile(0.05)
    q95 = r.quantile(0.95)
    q99 = r.quantile(0.99)

    # 下側テールと上側テールの大きさ比較
    tail_1pct_ratio = abs(q01) / q99 if q99 != 0 else np.nan
    tail_5pct_ratio = abs(q05) / q95 if q95 != 0 else np.nan

    # -----------------------------
    # 同じ閾値での超過確率比較
    # -----------------------------
    threshold_1sigma = std
    threshold_2sigma = 2 * std
    threshold_3sigma = 3 * std

    p_down_1sigma = (r < -threshold_1sigma).mean()
    p_up_1sigma = (r > threshold_1sigma).mean()

    p_down_2sigma = (r < -threshold_2sigma).mean()
    p_up_2sigma = (r > threshold_2sigma).mean()

    p_down_3sigma = (r < -threshold_3sigma).mean()
    p_up_3sigma = (r > threshold_3sigma).mean()

    # -----------------------------
    # 出力
    # -----------------------------
    print("=" * 70)
    print(f"{name} のリターン非対称性分析")
    print("=" * 70)

    print(f"データ数: {len(r)}")
    print(f"平均: {mean:.8f}")
    print(f"標準偏差: {std:.8f}")
    print(f"歪度 skewness: {skewness:.8f}")
    print()

    if skewness < 0:
        print("判定: 歪度が負なので、下落側に歪んだ分布です。")
    elif skewness > 0:
        print("判定: 歪度が正なので、上昇側に歪んだ分布です。")
    else:
        print("判定: 歪度はほぼ0で、左右対称に近いです。")

    print()
    print("上昇時・下落時の値幅比較")
    print(f"上昇時の平均値幅: {avg_up:.8f}")
    print(f"下落時の平均値幅: {avg_down:.8f}")
    print(f"下落平均値幅 / 上昇平均値幅: {avg_down_up_ratio:.4f}")
    print()

    print(f"上昇時の中央値値幅: {median_up:.8f}")
    print(f"下落時の中央値値幅: {median_down:.8f}")
    print(f"下落中央値値幅 / 上昇中央値値幅: {median_down_up_ratio:.4f}")
    print()

    print(f"最大上昇: {max_up:.8f}")
    print(f"最大下落の絶対値: {max_down:.8f}")
    print(f"最大下落 / 最大上昇: {max_down_up_ratio:.4f}")
    print()

    print("分位点ベースの下側・上側テール比較")
    print(f"1%分位点 q01: {q01:.8f}")
    print(f"99%分位点 q99: {q99:.8f}")
    print(f"|q01| / q99: {tail_1pct_ratio:.4f}")
    print()

    print(f"5%分位点 q05: {q05:.8f}")
    print(f"95%分位点 q95: {q95:.8f}")
    print(f"|q05| / q95: {tail_5pct_ratio:.4f}")
    print()

    print("同じ閾値での下落・上昇超過確率")
    print(f"P(r < -1σ): {p_down_1sigma:.6f}")
    print(f"P(r >  1σ): {p_up_1sigma:.6f}")
    print()

    print(f"P(r < -2σ): {p_down_2sigma:.6f}")
    print(f"P(r >  2σ): {p_up_2sigma:.6f}")
    print()

    print(f"P(r < -3σ): {p_down_3sigma:.6f}")
    print(f"P(r >  3σ): {p_up_3sigma:.6f}")
    print()

    # -----------------------------
    # 可視化1: ヒストグラム
    # -----------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(r, bins=80, density=True, alpha=0.6)
    plt.axvline(0, linestyle="--", label="zero")
    plt.axvline(q05, linestyle="--", label="5% quantile")
    plt.axvline(q95, linestyle="--", label="95% quantile")
    plt.title(f"{name}: Return Distribution and Asymmetry")
    plt.xlabel("return")
    plt.ylabel("density")
    plt.legend()
    plt.grid(True)
    plt.show()

    # -----------------------------
    # 可視化2: 上側・下側テールの比較
    # -----------------------------
    labels = ["1% tail", "5% tail"]
    downside = [abs(q01), abs(q05)]
    upside = [q99, q95]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, downside, width, label="Downside tail |q_low|")
    plt.bar(x + width / 2, upside, width, label="Upside tail q_high")
    plt.xticks(x, labels)
    plt.title(f"{name}: Downside vs Upside Tail Size")
    plt.ylabel("absolute return size")
    plt.legend()
    plt.grid(True)
    plt.show()

    result = {
        "name": name,
        "n": len(r),
        "mean": mean,
        "std": std,
        "skewness": skewness,
        "avg_up": avg_up,
        "avg_down_abs": avg_down,
        "avg_down_up_ratio": avg_down_up_ratio,
        "median_up": median_up,
        "median_down_abs": median_down,
        "median_down_up_ratio": median_down_up_ratio,
        "max_up": max_up,
        "max_down_abs": max_down,
        "max_down_up_ratio": max_down_up_ratio,
        "q01": q01,
        "q05": q05,
        "q95": q95,
        "q99": q99,
        "tail_1pct_ratio_abs_q01_over_q99": tail_1pct_ratio,
        "tail_5pct_ratio_abs_q05_over_q95": tail_5pct_ratio,
        "p_down_1sigma": p_down_1sigma,
        "p_up_1sigma": p_up_1sigma,
        "p_down_2sigma": p_down_2sigma,
        "p_up_2sigma": p_up_2sigma,
        "p_down_3sigma": p_down_3sigma,
        "p_up_3sigma": p_up_3sigma,
    }

    return result


# ============================================================
# 3. 実行
# ============================================================

result = analyze_return_asymmetry(r, name="S&P500 return")

# 結果をDataFrameで確認
result_df = pd.DataFrame([result])
print("=" * 70)
print("分析結果まとめ")
print("=" * 70)
print(result_df)

# %% Cell 1
# ============================================================
# 4. 生成データ mixed_sabr / mixed_brown のリターン非対称性確認
#    - 標準出力を log.txt に保存
#    - 集計結果を CSV に保存
#    - 各系列のグラフを PNG に保存
# ============================================================

from pathlib import Path
from contextlib import redirect_stdout
import re
import pandas as pd
import matplotlib.pyplot as plt


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


def safe_filename(name):
    """ファイル名に使いにくい文字を置換する。"""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", str(name))


def analyze_generated_return_asymmetry(csv_path, dataset_name, fig_dir=None):
    """
    mixedデータ内の mask*_sp500 / mask*_DGS10 を1系列ずつ非対称性分析する。

    - analyze_return_asymmetry() の標準出力は呼び出し側でlogに保存される
    - analyze_return_asymmetry() が作成したグラフは系列ごとにPNG保存する
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

            result = analyze_return_asymmetry(
                generated_df[col],
                name=title,
            )
        finally:
            plt.show = original_show

        # analyze_return_asymmetry() 内で作成されたグラフをすべて保存
        if fig_dir is not None:
            new_fig_nums = [
                num for num in plt.get_fignums()
                if num not in before_fig_nums
            ]
            figure_labels = ["return_distribution", "downside_vs_upside_tail"]

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
            result["dataset"] = dataset_name
            result["column"] = col
            generated_results.append(result)

    return pd.DataFrame(generated_results)


# ============================================================
# 出力先の設定
# ============================================================

output_dir = Path("results")
output_dir.mkdir(parents=True, exist_ok=True)

fig_dir = output_dir / "figures_generated_asymmetry"
fig_dir.mkdir(parents=True, exist_ok=True)

log_path = output_dir / "generated_asymmetry_log.txt"
csv_output_path = output_dir / "generated_asymmetry_summary.csv"


# ============================================================
# 分析実行
# ============================================================

with open(log_path, "w", encoding="utf-8") as f:
    with redirect_stdout(f):

        print("=" * 70)
        print("生成データ mixed_sabr / mixed_brown のリターン非対称性分析")
        print("=" * 70)

        mixed_sabr_path = resolve_data_path("mixed_sabr_masked.csv")
        mixed_brown_path = resolve_data_path("mixed_brown_masked.csv")

        print(f"mixed_sabr path : {mixed_sabr_path}")
        print(f"mixed_brown path: {mixed_brown_path}")
        print()

        mixed_sabr_asymmetry_results = analyze_generated_return_asymmetry(
            mixed_sabr_path,
            "mixed_sabr",
            fig_dir=fig_dir,
        )

        mixed_brown_asymmetry_results = analyze_generated_return_asymmetry(
            mixed_brown_path,
            "mixed_brown",
            fig_dir=fig_dir,
        )

        mixed_asymmetry_results = pd.concat(
            [mixed_sabr_asymmetry_results, mixed_brown_asymmetry_results],
            ignore_index=True,
        )

        summary_columns = [
            "dataset",
            "column",
            "n",
            "mean",
            "std",
            "skewness",
            "avg_down_up_ratio",
            "median_down_up_ratio",
            "max_down_up_ratio",
            "tail_1pct_ratio_abs_q01_over_q99",
            "tail_5pct_ratio_abs_q05_over_q95",
            "p_down_1sigma",
            "p_up_1sigma",
            "p_down_2sigma",
            "p_up_2sigma",
            "p_down_3sigma",
            "p_up_3sigma",
        ]

        summary_df = mixed_asymmetry_results[summary_columns]

        print()
        print("=" * 70)
        print("生成データ リターン非対称性分析結果まとめ")
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
