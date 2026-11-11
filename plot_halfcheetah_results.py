"""
HalfCheetah-v5 実験結果のプロットスクリプト
====================================================

対象は run_halfcheetah_experiments.py の結果（遅延アンカー vs 素の SAC、
HalfCheetah-v5 / batch_size=256 / 3M steps / 5 seeds）。

使い方（リポジトリのルートで）:
    python plot_halfcheetah_results.py

入力 (npz):
    ./npz_logs_delayed_anchor_halfcheetah/delayed_anchor_halfcheetah_seed{0..4}.npz
    ./npz_logs_pure_sac_halfcheetah/pure_sac_halfcheetah_seed{0..4}.npz
    （各 npz は timesteps / returns を含むこと）

出力 (./plots_halfcheetah/ に3枚):
    1. learning_curves_overlay.png
       両手法の学習曲線（シード平均 ± 標準偏差）を重ねた図。論文の主図用。
    2. learning_curves_by_seed.png
       手法ごとにシード別の曲線を全部描いた図。崩壊シードの有無が一目で分かる。
    3. final_window.png
       シード別の最終区間（2.5M〜3.0M）平均の棒グラフ。
       崩壊ライン 4,000 と IQM（中3つ平均）のマーカー付き。

あわせて、判定に使う集計値（シード別最終区間平均、平均・中央値・IQM、
崩壊数、AUC）を標準出力に表示する。定義は halfcheetah_judgment.py と同一:
    - 最終区間平均: 2.5M〜3.0M の評価リターンの平均
    - IQM: 上下を除いた中3つの平均
    - 崩壊: 最終区間平均が 4,000 未満
    - AUC: 学習曲線全体の台形積分（報告用。成功条件には使わない）

注意:
    - 図中の文字は英語（計算機側に日本語フォントが無くても文字化けしないため）。
    - 曲線の平滑化（移動平均）は表示専用で、集計値には生の評価値を使う。
    - npz が不足している場合は、読めた分だけで図を作り、不足分を表示する。
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINAL_START = 2_500_000
COLLAPSE = 4_000

DELAYED_COLOR = "tab:blue"
SAC_COLOR = "tab:orange"


def load_curves(log_dir, filename_pattern, seeds):
    """npz を読み、(timesteps, returns) のリストと読めたファイル名を返す。"""
    curves, loaded = [], []
    for s in seeds:
        path = os.path.join(log_dir, filename_pattern.format(seed=s))
        if not os.path.exists(path):
            print(f"  [不足] {path} がありません")
            continue
        d = np.load(path)
        curves.append((np.asarray(d["timesteps"], dtype=float),
                       np.asarray(d["returns"], dtype=float)))
        loaded.append(path)
        print(f"  [読込] {path}")
    return curves, loaded


def common_grid(curves):
    """全シード共通の評価グリッド（最初のシードの timesteps を使う）。"""
    return curves[0][0]


def interp_all(curves, grid):
    """各シードの returns を共通グリッドへ線形補間し、2次元配列にする。"""
    return np.array([np.interp(grid, ts, rs) for ts, rs in curves])


def smooth(y, window):
    """表示専用の移動平均。window は評価点の個数（奇数に丸める）。"""
    if window is None or window <= 1:
        return y
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def final_mean(timesteps, returns):
    mask = timesteps >= FINAL_START
    return float(np.mean(returns[mask]))


def auc(timesteps, returns):
    trapz = getattr(np, "trapezoid", None) or np.trapz
    order = np.argsort(timesteps)
    return float(trapz(returns[order], timesteps[order]))


def iqm(values):
    v = np.sort(np.asarray(values, dtype=float))
    k = len(v) // 4
    core = v[k: len(v) - k] if k else v
    return float(np.mean(core))


def summarize(name, curves):
    finals = [final_mean(ts, rs) for ts, rs in curves]
    aucs = [auc(ts, rs) for ts, rs in curves]
    print(f"\n[{name}] シード別 最終区間平均: "
          f"{[round(f) for f in finals]}")
    print(f"  平均 {np.mean(finals):,.0f} / 中央値 {np.median(finals):,.0f} / "
          f"IQM {iqm(finals):,.0f} / 崩壊 ({COLLAPSE:,}未満) "
          f"{int(np.sum(np.array(finals) < COLLAPSE))} シード")
    print(f"  AUC: シード別 {[round(a) for a in aucs]} / "
          f"平均 {np.mean(aucs):,.0f}")
    return finals, aucs


def plot_overlay(grid, d_mat, s_mat, window, out_path):
    """図1: 両手法の平均 ± 標準偏差を重ねる。"""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mat, color, label in ((d_mat, DELAYED_COLOR, "Delayed anchor (proposed)"),
                              (s_mat, SAC_COLOR, "Pure SAC")):
        mean = smooth(mat.mean(axis=0), window)
        std = smooth(mat.std(axis=0), window)
        ax.plot(grid / 1e6, mean, color=color, lw=2, label=label)
        ax.fill_between(grid / 1e6, mean - std, mean + std,
                        color=color, alpha=0.2)
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Episodic return (eval, mean of 5 episodes)")
    ax.set_title("HalfCheetah-v5: learning curves (5 seeds, mean ± std)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_by_seed(grid, d_mat, s_mat, window, out_path):
    """図2: 手法ごとにシード別の曲線を全部描く。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, mat, color, title in (
            (axes[0], d_mat, DELAYED_COLOR, "Delayed anchor (proposed)"),
            (axes[1], s_mat, SAC_COLOR, "Pure SAC")):
        for i, row in enumerate(mat):
            ax.plot(grid / 1e6, smooth(row, window), lw=1, alpha=0.55,
                    label=f"seed {i}")
        ax.plot(grid / 1e6, smooth(mat.mean(axis=0), window),
                color="black", lw=2, label="mean")
        ax.axhline(COLLAPSE, color="gray", ls="--", lw=1)
        ax.set_xlabel("Timesteps (M)")
        ax.set_title(title)
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Episodic return (eval, mean of 5 episodes)")
    fig.suptitle("HalfCheetah-v5: per-seed learning curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_final_window(d_finals, s_finals, out_path):
    """図3: シード別の最終区間平均（棒）+ 崩壊ライン + IQM マーカー。"""
    seeds = np.arange(len(d_finals))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(seeds - width / 2, d_finals, width,
           color=DELAYED_COLOR, label="Delayed anchor (proposed)")
    ax.bar(seeds + width / 2, s_finals, width,
           color=SAC_COLOR, label="Pure SAC")
    ax.axhline(COLLAPSE, color="red", ls="--", lw=1.2,
               label="Collapse threshold (4,000)")
    ax.scatter([len(seeds) - 0.3], [iqm(d_finals)], marker="D", s=90,
               color=DELAYED_COLOR, edgecolor="black", zorder=5,
               label="IQM (delayed)")
    ax.scatter([len(seeds) - 0.3], [iqm(s_finals)], marker="D", s=90,
               color=SAC_COLOR, edgecolor="black", zorder=5, label="IQM (SAC)")
    ax.set_xlim(-0.7, len(seeds) + 0.1)
    ax.set_xticks(list(seeds))
    ax.set_xticklabels([f"seed {i}" for i in seeds])
    ax.set_ylabel("Final-window mean return (2.5M-3.0M)")
    ax.set_title("HalfCheetah-v5: final performance per seed")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="HalfCheetah 結果のプロット")
    ap.add_argument("--delayed-dir", default="./npz_logs_delayed_anchor_halfcheetah")
    ap.add_argument("--sac-dir", default="./npz_logs_pure_sac_halfcheetah")
    ap.add_argument("--out-dir", default="./plots_halfcheetah")
    ap.add_argument("--smooth", type=int, default=11,
                    help="表示用の移動平均の幅（評価点の個数、既定 11 = 55k steps）")
    args = ap.parse_args()

    seeds = range(5)
    print("== 遅延アンカー ==")
    d_curves, _ = load_curves(args.delayed_dir,
                              "delayed_anchor_halfcheetah_seed{seed}.npz", seeds)
    print("== 素の SAC ==")
    s_curves, _ = load_curves(args.sac_dir,
                              "pure_sac_halfcheetah_seed{seed}.npz", seeds)
    if not d_curves or not s_curves:
        print("どちらかの手法の npz が1つも読めませんでした。終了します。")
        return

    d_finals, _ = summarize("遅延アンカー", d_curves)
    s_finals, _ = summarize("素の SAC", s_curves)

    os.makedirs(args.out_dir, exist_ok=True)
    print("\n== 図の作成 ==")
    # 共通グリッドは遅延 seed 0 のものを使う（全ランで評価周期は同一のはず）
    grid = common_grid(d_curves)
    d_mat = interp_all(d_curves, grid)
    s_mat = interp_all(s_curves, grid)
    plot_overlay(grid, d_mat, s_mat, args.smooth,
                 os.path.join(args.out_dir, "learning_curves_overlay.png"))
    plot_by_seed(grid, d_mat, s_mat, args.smooth,
                 os.path.join(args.out_dir, "learning_curves_by_seed.png"))
    if len(d_finals) == 5 and len(s_finals) == 5:
        plot_final_window(d_finals, s_finals,
                          os.path.join(args.out_dir, "final_window.png"))
    else:
        print("  [注意] 最終区間の図は両手法 5 シード揃った場合のみ作成します")


if __name__ == "__main__":
    main()
