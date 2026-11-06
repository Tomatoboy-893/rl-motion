"""
遅延アンカー版 ADR の結果プロット用スクリプト
================================================

run_delayed_anchor_experiments.py が保存した npz を読み込んで、
確認用の図を3枚作ります。計算機（SSH先）でそのまま実行できます。

使い方
------
    python plot_delayed_anchor_results.py
    # → ./plots/ フォルダに PNG が3枚できます

    # フォルダを変えたいとき
    python plot_delayed_anchor_results.py \
        --delayed-dir ./npz_logs_delayed_anchor_bs64 \
        --sac-dir ./npz_logs_pure_sac_bs64 \
        --out-dir ./plots

作られる図
----------
1. returns_comparison.png
   評価リターンの比較（シードが複数あれば mean ± std の帯付き）。
   pure_sac の npz が無ければ提案手法だけ描きます（試し回し用）。

2. diagnostics_comparison.png
   学習の診断図（2x2）。前回の Humanoid 実験の図と同じ並びです。
   - 左上: 方策エントロピー H(π)
   - 右上: α（※ log ではなく α そのもの。軸ラベルも正直に α と表記）
   - 左下: KL(π || アンカー) と目標値 target_kl（合計）の線
   - 右下: β（※ log ではなく β そのもの。ただし軸は対数スケール。
           線形軸だと小さい値の動きが見えないため）

3. kl_beta_detail.png
   試し回しの確認専用の拡大図（1x2）。
   - 左: KL の内訳（合計 / 平均項 / σ項）＋ target_kl の線
         → 「どちらの項が KL を作っているか」が分かります
   - 右: β（対数軸）＋ 下限 beta_min の線
         → 「β が下限に張り付いていないか（＝装置が死んでいないか）」
           が分かります

前提・注意
----------
- npz のファイル名は *_seed{番号}.npz の形式を想定し、
  同じフォルダ内のファイルを seed ごとに自動で拾います。
- 学習ログ（kl_values など）は点数が多い（最大で数百万点）ため、
  描画用に間引いてから平滑化しています。元の値は変えていません。
- target_kl の線は「0.05/次元 × 17次元 = 0.85（合計）」で固定表示します。
  実験側の target_kl を変えたときは、このファイルの TARGET_KL_TOTAL
  も合わせて変えてください。
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")  # SSH先など画面のない環境でも動くようにする設定
import matplotlib.pyplot as plt


# ============================================================
# 設定（実験側と合わせる場所）
# ============================================================
ACT_DIM = 17                 # Humanoid-v5 の行動次元数
TARGET_KL_PER_DIM = 0.05     # 実験側の target_kl（1次元あたり）
TARGET_KL_TOTAL = ACT_DIM * TARGET_KL_PER_DIM  # 図に引く目標線（合計 = 0.85）
BETA_MIN = 1e-4              # 実験側の beta_min（β の下限）

# 色は前回の Humanoid の図と揃える（青 = 提案手法、橙 = 素の SAC）
COLOR_OURS = "#1f77b4"
COLOR_SAC = "#ff7f0e"


# ============================================================
# 小物ユーティリティ
# ============================================================
def moving_average(x, window):
    """移動平均で平滑化する。window が 1 以下ならそのまま返す。"""
    x = np.asarray(x, dtype=float)
    if window is None or window <= 1 or len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def downsample(x, y, max_points=5000):
    """点数が多すぎる系列を、描画用に最大 max_points 点まで間引く。"""
    if len(x) <= max_points:
        return x, y
    idx = np.linspace(0, len(x) - 1, max_points).astype(int)
    return np.asarray(x)[idx], np.asarray(y)[idx]


def load_seed_files(log_dir):
    """フォルダ内の npz を seed 番号つきで読み込む。

    戻り値: {seed番号: npzの辞書} 。フォルダが無い・空なら {} を返す。
    """
    result = {}
    if not log_dir or not os.path.isdir(log_dir):
        return result
    for path in sorted(glob.glob(os.path.join(log_dir, "*.npz"))):
        m = re.search(r"seed(\d+)", os.path.basename(path))
        seed = int(m.group(1)) if m else len(result)
        result[seed] = dict(np.load(path))
    return result


def stack_series(seed_data, key):
    """全 seed の系列を、最小の長さに切りそろえて (seed数, 長さ) に重ねる。

    key を持たない seed や、系列が空の場合は None を返す。
    """
    series = []
    for seed in sorted(seed_data):
        data = seed_data[seed]
        if key in data and len(data[key]) > 0:
            series.append(np.asarray(data[key], dtype=float))
    if not series:
        return None
    min_len = min(len(s) for s in series)
    return np.stack([s[:min_len] for s in series])


def plot_mean_std(ax, stacked, color, label, smooth=1):
    """(seed数, 長さ) の行列を、平均の線＋標準偏差の帯で描く。

    seed が1つだけのときは帯を描かず、線だけにする。
    x 軸は「更新ステップ数（= ログのインデックス）」。
    """
    if stacked is None:
        return
    mean = stacked.mean(axis=0)
    x = np.arange(len(mean))
    if smooth > 1:
        mean_s = moving_average(mean, smooth)
        x_s = np.arange(len(mean_s)) + smooth // 2
    else:
        mean_s, x_s = mean, x
    x_s, mean_s = downsample(x_s, mean_s)
    ax.plot(x_s, mean_s, color=color, label=label, linewidth=1.2)
    if stacked.shape[0] > 1:
        std = stacked.std(axis=0)
        lo, hi = mean - std, mean + std
        if smooth > 1:
            lo = moving_average(lo, smooth)
            hi = moving_average(hi, smooth)
            xs = np.arange(len(lo)) + smooth // 2
        else:
            xs = x
        xs, lo = downsample(xs, lo)
        _, hi = downsample(np.arange(len(hi)), hi)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.2)


# ============================================================
# 図1: 評価リターンの比較
# ============================================================
def plot_returns(delayed, sac, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    for seed_data, color, label in [
        (delayed, COLOR_OURS, "Delayed-Anchor Prior SAC (Ours)"),
        (sac, COLOR_SAC, "Vanilla SAC"),
    ]:
        # seed ごとに (timesteps, returns) を集めて平均と帯を作る
        curves = []
        for seed in sorted(seed_data):
            d = seed_data[seed]
            if "returns" in d and len(d["returns"]) > 0:
                curves.append((np.asarray(d["timesteps"], dtype=float),
                                 np.asarray(d["returns"], dtype=float)))
        if not curves:
            continue
        min_len = min(len(r) for _, r in curves)
        t = curves[0][0][:min_len]
        R = np.stack([r[:min_len] for _, r in curves])
        mean = R.mean(axis=0)
        ax.plot(t, mean, color=color, label=label, linewidth=1.5)
        if R.shape[0] > 1:
            std = R.std(axis=0)
            ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2)

    ax.set_title("Humanoid-v5: Evaluation Returns Comparison")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "returns_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved: {path}")


# ============================================================
# 図2: 診断図（2x2）
# ============================================================
def plot_diagnostics(delayed, sac, out_dir, smooth):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Humanoid-v5: Training Diagnostics & Dynamic Parameters", fontsize=14)

    # --- 左上: 方策エントロピー ---
    ax = axes[0, 0]
    plot_mean_std(ax, stack_series(delayed, "pi_entropies"), COLOR_OURS,
                  "Delayed-Anchor Prior (Ours)", smooth)
    plot_mean_std(ax, stack_series(sac, "pi_entropies"), COLOR_SAC,
                  "Vanilla SAC", smooth)
    ax.set_title("Policy Entropy ($H(\\pi)$)")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("Entropy")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # --- 右上: α（log ではなく α そのもの） ---
    ax = axes[0, 1]
    plot_mean_std(ax, stack_series(delayed, "alpha_logs"), COLOR_OURS,
                  "Delayed-Anchor Prior (Ours)", smooth)
    plot_mean_std(ax, stack_series(sac, "alpha_logs"), COLOR_SAC,
                  "Vanilla SAC", smooth)
    ax.set_title("Temperature Parameter ($\\alpha$)")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("$\\alpha$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # --- 左下: KL と目標値 ---
    ax = axes[1, 0]
    plot_mean_std(ax, stack_series(delayed, "kl_values"), "green",
                  "KL($\\pi$ || Anchor)", smooth)
    ax.axhline(TARGET_KL_TOTAL, color="red", linestyle="--", linewidth=1.0,
               label=f"target KL (total) = {TARGET_KL_TOTAL:.2f}")
    ax.set_title("KL Divergence from Anchor Prior")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("KL Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # --- 右下: β（値はそのもの、軸は対数） ---
    ax = axes[1, 1]
    plot_mean_std(ax, stack_series(delayed, "beta_logs"), "purple",
                  "$\\beta$ (KL Coeff)", smooth)
    ax.axhline(BETA_MIN, color="red", linestyle="--", linewidth=1.0,
               label=f"beta_min = {BETA_MIN:g}")
    ax.set_yscale("log")
    ax.set_title("Adaptive KL Weight ($\\beta$, log scale)")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("$\\beta$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(out_dir, "diagnostics_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved: {path}")


# ============================================================
# 図3: KL内訳とβの拡大図（試し回しの確認用）
# ============================================================
def plot_kl_beta_detail(delayed, out_dir, smooth):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- 左: KL の内訳（合計 / 平均項 / σ項） ---
    ax = axes[0]
    plot_mean_std(ax, stack_series(delayed, "kl_values"), "green",
                  "KL total", smooth)
    plot_mean_std(ax, stack_series(delayed, "kl_mean_terms"), "blue",
                  "mean term", smooth)
    plot_mean_std(ax, stack_series(delayed, "kl_std_terms"), "orange",
                  "std term", smooth)
    ax.axhline(TARGET_KL_TOTAL, color="red", linestyle="--", linewidth=1.0,
               label=f"target KL (total) = {TARGET_KL_TOTAL:.2f}")
    ax.set_title("KL Breakdown: mean term vs std term")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("KL Value")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    # --- 右: β（対数軸）＋ 下限の線 ---
    ax = axes[1]
    plot_mean_std(ax, stack_series(delayed, "beta_logs"), "purple",
                  "$\\beta$", smooth)
    ax.axhline(BETA_MIN, color="red", linestyle="--", linewidth=1.0,
               label=f"beta_min = {BETA_MIN:g}")
    ax.axhline(0.01, color="gray", linestyle=":", linewidth=1.0,
               label="initial beta = 0.01")
    ax.set_yscale("log")
    ax.set_title("Beta (log scale): is the guard alive?")
    ax.set_xlabel("Update Steps")
    ax.set_ylabel("$\\beta$")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    path = os.path.join(out_dir, "kl_beta_detail.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved: {path}")


# ============================================================
# メイン
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="遅延アンカー版 ADR の結果をプロットする")
    parser.add_argument("--delayed-dir", default="./npz_logs_delayed_anchor_bs64",
                        help="提案手法の npz フォルダ")
    parser.add_argument("--sac-dir", default="./npz_logs_pure_sac_bs64",
                        help="素の SAC の npz フォルダ（無ければ提案手法だけ描く）")
    parser.add_argument("--out-dir", default="./plots", help="図の出力先フォルダ")
    parser.add_argument("--smooth", type=int, default=101,
                        help="学習ログの平滑化ウィンドウ（移動平均の点数）")
    args = parser.parse_args()

    delayed = load_seed_files(args.delayed_dir)
    sac = load_seed_files(args.sac_dir)

    if not delayed and not sac:
        print("npz が見つかりませんでした。--delayed-dir / --sac-dir のパスを確認してください。")
        return

    print(f"delayed_anchor: {len(delayed)} seed(s), pure_sac: {len(sac)} seed(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_returns(delayed, sac, args.out_dir)
    plot_diagnostics(delayed, sac, args.out_dir, args.smooth)
    if delayed:
        plot_kl_beta_detail(delayed, args.out_dir, args.smooth)


if __name__ == "__main__":
    main()
