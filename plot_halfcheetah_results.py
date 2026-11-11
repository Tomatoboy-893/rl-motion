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

出力 (./plots_halfcheetah/ に5枚):
    1. learning_curves_overlay.png
       両手法の学習曲線（シード平均 ± 標準偏差）を重ねた図。論文の主図用。
    2. learning_curves_by_seed.png
       手法ごとにシード別の曲線を全部描いた図。崩壊シードの有無が一目で分かる。
    3. final_window.png
       シード別の最終区間（2.5M〜3.0M）平均の棒グラフ。
       崩壊ライン 4,000 と IQM（中3つ平均）のマーカー付き。
    4. diagnostics_delayed.png（遅延アンカー側の診断、2x2）
       KL（上限適用後、目標 0.30 の線付き）／ β の推移（対数軸）／
       KL の内訳（平均項と σ項）／ 生 KL（上限適用前、対数軸）。
       「β が生きているか」「KL が目標に張り付くか」「平均項が支配的か」
       を確認するための図。
    5. diagnostics_entropy_alpha.png（両手法共通の診断、1x2）
       方策エントロピーと α の推移の重ね図。
       σ の狭まり方（序盤の固着の分析）に使う。

診断ログ（kl_values など）は勾配ステップごとに記録されるため点数が
非常に多い（約300万点）。表示用に 600 区間へ平均化してから描く
（生 KL のパネルだけはスパイクが見えるよう区間最大値を使う）。
診断の横軸は「ログの記録位置 ≒ 学習ステップ数」で近似している。

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


def load_diag(log_dir, filename_pattern, seeds, keys):
    """診断ログを読む。戻り値は {key: [シードごとの配列]}。
    キーが無い npz はそのキーの分だけ読み飛ばし、表示する。"""
    out = {k: [] for k in keys}
    for s in seeds:
        path = os.path.join(log_dir, filename_pattern.format(seed=s))
        if not os.path.exists(path):
            continue
        d = np.load(path)
        for k in keys:
            if k in d:
                out[k].append(np.asarray(d[k], dtype=float))
            else:
                print(f"  [不足] {path} に {k} がありません")
    return out


def bin_reduce(y, n_bins=600, how="mean"):
    """長い時系列を表示用に n_bins 区間へ圧縮する（how: mean / max）。"""
    y = np.asarray(y, dtype=float)
    if len(y) <= n_bins:
        return y
    trim = (len(y) // n_bins) * n_bins
    blocks = y[:trim].reshape(n_bins, -1)
    return blocks.mean(axis=1) if how == "mean" else blocks.max(axis=1)


def diag_x(n_points, total_steps, n_bins=600):
    """診断ログの横軸（記録位置を学習ステップ数に近似）。"""
    n = min(n_points, n_bins) if n_points > n_bins else n_points
    return np.linspace(0, total_steps, n)


def plot_diagnostics_delayed(diag, total_steps, target_kl, out_path):
    """図4: 遅延アンカー側の診断 2x2（KL / β / KL内訳 / 生KL）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def panel(ax, key, title, ylabel, logy=False, how="mean", hline=None):
        series = diag.get(key, [])
        if not series:
            ax.set_title(f"{title} (no data)")
            return
        binned = [bin_reduce(y, how=how) for y in series]
        for i, b in enumerate(binned):
            ax.plot(diag_x(len(series[i]), total_steps)[:len(b)] / 1e6, b,
                    lw=0.8, alpha=0.55, label=f"seed {i}")
        mat = np.array([b[:min(len(x) for x in binned)] for b in binned])
        xs = diag_x(len(series[0]), total_steps)[:mat.shape[1]] / 1e6
        ax.plot(xs, mat.mean(axis=0), color="black", lw=1.8, label="mean")
        if hline is not None:
            ax.axhline(hline, color="red", ls="--", lw=1.2,
                       label=f"target ({hline:g})")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("Training steps (M, approx. from log index)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    panel(axes[0, 0], "kl_values", "KL divergence (capped, used in loss)",
          "KL (total)", hline=target_kl)
    panel(axes[0, 1], "beta_logs", "Beta (ADR coefficient)", "beta",
          logy=True)
    # KL の内訳はシード平均同士の比較で十分なので専用に描く
    ax = axes[1, 0]
    mean_terms, std_terms = diag.get("kl_mean_terms", []), diag.get("kl_std_terms", [])
    if mean_terms and std_terms:
        bm = np.array([bin_reduce(y) for y in mean_terms]).mean(axis=0)
        bs = np.array([bin_reduce(y) for y in std_terms]).mean(axis=0)
        n = min(len(bm), len(bs))
        xs = diag_x(len(mean_terms[0]), total_steps)[:n] / 1e6
        ax.plot(xs, bm[:n], lw=1.8, label="mean term (across-seed mean)")
        ax.plot(xs, bs[:n], lw=1.8, label="std term (across-seed mean)")
    ax.set_xlabel("Training steps (M, approx. from log index)")
    ax.set_ylabel("KL term (total)")
    ax.set_title("KL decomposition")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    panel(axes[1, 1], "kl_raw_values", "Raw KL (before cap, bin max)",
          "KL raw (total)", logy=True, how="max")
    fig.suptitle("Delayed anchor: diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_entropy_alpha(d_diag, s_diag, total_steps, out_path):
    """図5: 両手法のエントロピーと α の重ね図。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, ylabel in (
            (axes[0], "pi_entropies", "Policy entropy (pre-tanh Gaussian)",
             "Entropy"),
            (axes[1], "alpha_logs", "Alpha (entropy coefficient)", "alpha")):
        for diag, color, label in ((d_diag, DELAYED_COLOR, "Delayed anchor"),
                                   (s_diag, SAC_COLOR, "Pure SAC")):
            series = diag.get(key, [])
            if not series:
                continue
            binned = [bin_reduce(y) for y in series]
            for b in binned:
                ax.plot(diag_x(len(b), total_steps, n_bins=len(b)) / 1e6, b,
                        color=color, lw=0.7, alpha=0.4)
            mat = np.array([b[:min(len(x) for x in binned)] for b in binned])
            xs = diag_x(mat.shape[1], total_steps, n_bins=mat.shape[1]) / 1e6
            ax.plot(xs, mat.mean(axis=0), color=color, lw=2, label=label)
        ax.set_xlabel("Training steps (M, approx. from log index)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Both methods: entropy and alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


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
    ap.add_argument("--target-kl", type=float, default=0.30,
                    help="KL の目標値（合計）。既定 0.30 = 0.05/次元 × 6次元")
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

    # 診断図（npz 内の診断ログがある場合のみ）
    total_steps = float(grid.max())
    print("\n== 診断ログの読み込み ==")
    d_diag = load_diag(args.delayed_dir,
                       "delayed_anchor_halfcheetah_seed{seed}.npz", seeds,
                       ["kl_values", "kl_raw_values", "kl_mean_terms",
                        "kl_std_terms", "beta_logs", "pi_entropies",
                        "alpha_logs"])
    s_diag = load_diag(args.sac_dir,
                       "pure_sac_halfcheetah_seed{seed}.npz", seeds,
                       ["pi_entropies", "alpha_logs"])
    if d_diag.get("kl_values"):
        plot_diagnostics_delayed(
            d_diag, total_steps, args.target_kl,
            os.path.join(args.out_dir, "diagnostics_delayed.png"))
    else:
        print("  [注意] 遅延側に kl_values が無いため診断図4は作りません")
    if d_diag.get("pi_entropies") and s_diag.get("pi_entropies"):
        plot_entropy_alpha(
            d_diag, s_diag, total_steps,
            os.path.join(args.out_dir, "diagnostics_entropy_alpha.png"))
    else:
        print("  [注意] エントロピーが揃わないため診断図5は作りません")


if __name__ == "__main__":
    main()
