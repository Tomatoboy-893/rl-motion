"""
HalfCheetah-v5 3手法比較のプロットスクリプト
====================================================

対象は同一条件（HalfCheetah-v5 / batch_size=256 / 3M steps / 5 seeds）で
そろえた3手法:
    - 遅延アンカー版（提案手法、run_halfcheetah_experiments.py）
    - 先輩版 FixedPrior（run_fixedprior_halfcheetah_experiments.py）
    - 素の SAC（run_halfcheetah_experiments.py）

使い方（リポジトリのルートで）:
    python plot_three_way_results.py

入力 (npz):
    ./npz_logs_delayed_anchor_halfcheetah/delayed_anchor_halfcheetah_seed{0..4}.npz
    ./npz_logs_fixed_prior_halfcheetah/fixed_prior_halfcheetah_seed{0..4}.npz
    ./npz_logs_pure_sac_halfcheetah/pure_sac_halfcheetah_seed{0..4}.npz

出力 (./plots_three_way/ に最大6枚):
    1. learning_curves_overlay.png
       3手法の学習曲線（シード平均 ± 標準偏差）の重ね図。論文の主図候補。
    2. learning_curves_by_seed.png
       手法ごとにシード別の曲線を全部描いた図。崩壊シードの有無が一目で分かる。
    3. final_window.png
       シード別の最終区間（2.5M〜3.0M）平均の棒グラフ（3手法）。
       崩壊ライン 4,000 と IQM（中3つ平均）のマーカー付き。
    4. diagnostics_fixed_prior.png（先輩版の診断、2x2）
       KL（目標 6.0 の線付き）／ β の推移（対数軸）／
       方策エントロピー ／ α の推移。
       「KL が目標 6.0 付近にいるか」「β が消えていないか」を確認する図。
    5. diagnostics_kl_beta_comparison.png（ADR 2手法の KL・β 比較、2x2）
       遅延アンカーの KL（目標 0.30）／ 先輩版の KL（目標 6.0）／
       遅延アンカーの β ／ 先輩版の β。目標値が違うのでパネルは分けてある。
    6. diagnostics_entropy_alpha.png（3手法共通の診断、1x2）
       方策エントロピーと α の推移の重ね図。

あわせて、集計値（シード別最終区間平均、平均・中央値・IQM・IQMの
ブートストラップ95%信頼区間、崩壊数、AUC）と、ADR 2手法の KL・β の
数値要約（全体平均と終盤10%の平均）を標準出力に表示する。
定義は halfcheetah_judgment.py / plot_halfcheetah_results.py と同一:
    - 最終区間平均: 2.5M〜3.0M の評価リターンの平均
    - IQM: 上下を除いた中3つの平均
    - 崩壊: 最終区間平均が 4,000 未満
    - AUC: 学習曲線全体の台形積分（報告用。判定条件には使わない）

注意:
    - 先輩版との比較は記述的な比較であり、このスクリプトは新しい
      成功基準による判定は行わない（数値の表示のみ）。
    - 図中の文字は英語（計算機側に日本語フォントが無くても文字化けしないため）。
    - 曲線の平滑化（移動平均）は表示専用で、集計値には生の評価値を使う。
    - 診断ログは勾配ステップごとに記録され点数が非常に多い（約300万点）。
      表示用に 600 区間へ平均化してから描く。診断の横軸は
      「ログの記録位置 ≒ 学習ステップ数」で近似している。
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

METHODS = [
    {
        "key": "delayed",
        "label": "Delayed anchor (proposed)",
        "short": "遅延アンカー",
        "color": "tab:blue",
        "dir_arg": "delayed_dir",
        "pattern": "delayed_anchor_halfcheetah_seed{seed}.npz",
        "target_kl": 0.30,
    },
    {
        "key": "fixed",
        "label": "Fixed prior (senior ADR)",
        "short": "先輩版 FixedPrior",
        "color": "tab:green",
        "dir_arg": "fixed_dir",
        "pattern": "fixed_prior_halfcheetah_seed{seed}.npz",
        "target_kl": 6.0,
    },
    {
        "key": "sac",
        "label": "Pure SAC",
        "short": "素の SAC",
        "color": "tab:orange",
        "dir_arg": "sac_dir",
        "pattern": "pure_sac_halfcheetah_seed{seed}.npz",
        "target_kl": None,
    },
]


def load_curves(log_dir, filename_pattern, seeds):
    """npz を読み、(timesteps, returns) のリストを返す。"""
    curves = []
    for s in seeds:
        path = os.path.join(log_dir, filename_pattern.format(seed=s))
        if not os.path.exists(path):
            print(f"  [不足] {path} がありません")
            continue
        d = np.load(path)
        curves.append((np.asarray(d["timesteps"], dtype=float),
                       np.asarray(d["returns"], dtype=float)))
        print(f"  [読込] {path}")
    return curves


def load_diag(log_dir, filename_pattern, seeds, keys):
    """診断ログを読む。戻り値は {key: [シードごとの配列]}。"""
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


def interp_all(curves, grid):
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


def iqm_bootstrap_ci(values, n_boot=20000, seed=12345):
    """IQM のブートストラップ95%信頼区間（シードのリサンプリング）。"""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    samples = v[idx]
    samples.sort(axis=1)
    k = len(v) // 4
    core = samples[:, k: len(v) - k] if k else samples
    boots = core.mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def bin_reduce(y, n_bins=600, how="mean"):
    """長い時系列を表示用に n_bins 区間へ圧縮する。"""
    y = np.asarray(y, dtype=float)
    if len(y) <= n_bins:
        return y
    trim = (len(y) // n_bins) * n_bins
    blocks = y[:trim].reshape(n_bins, -1)
    return blocks.mean(axis=1) if how == "mean" else blocks.max(axis=1)


def diag_x(n_points, total_steps, n_bins=600):
    n = min(n_points, n_bins) if n_points > n_bins else n_points
    return np.linspace(0, total_steps, n)


def summarize(name, curves):
    finals = [final_mean(ts, rs) for ts, rs in curves]
    aucs = [auc(ts, rs) for ts, rs in curves]
    print(f"\n[{name}] シード別 最終区間平均: "
          f"{[round(f) for f in finals]}")
    if len(finals) >= 2:
        lo, hi = iqm_bootstrap_ci(finals)
        print(f"  平均 {np.mean(finals):,.0f} / 中央値 {np.median(finals):,.0f} / "
              f"IQM {iqm(finals):,.0f} (95%CI {lo:,.0f}〜{hi:,.0f}) / "
              f"崩壊 ({COLLAPSE:,}未満) "
              f"{int(np.sum(np.array(finals) < COLLAPSE))} シード")
    print(f"  AUC: シード別 {[round(a) for a in aucs]} / "
          f"平均 {np.mean(aucs):,.0f}")
    return finals, aucs


def summarize_diag(name, diag):
    """KL・β の数値要約（シードごとの全体平均と終盤10%平均を、さらに平均）。"""
    for key, label in (("kl_values", "KL"), ("beta_logs", "beta")):
        series = diag.get(key, [])
        if not series:
            continue
        all_means = [float(np.mean(y)) for y in series]
        tail_means = [float(np.mean(y[int(len(y) * 0.9):])) for y in series]
        print(f"  [{name}] {label}: シード別 全体平均 "
              f"{[round(x, 3) for x in all_means]} / 終盤10%平均 "
              f"{[round(x, 3) for x in tail_means]}")


def plot_overlay(grid, mats, window, out_path):
    """図1: 手法ごとの平均 ± 標準偏差を重ねる。mats は (method, 行列) のリスト。"""
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for m, mat in mats:
        mean = smooth(mat.mean(axis=0), window)
        std = smooth(mat.std(axis=0), window)
        ax.plot(grid / 1e6, mean, color=m["color"], lw=2, label=m["label"])
        ax.fill_between(grid / 1e6, mean - std, mean + std,
                        color=m["color"], alpha=0.18)
    ax.set_xlabel("Timesteps (M)")
    ax.set_ylabel("Episodic return (eval, mean of 5 episodes)")
    ax.set_title("HalfCheetah-v5: learning curves (5 seeds, mean ± std)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_by_seed(grid, mats, window, out_path):
    """図2: 手法ごとにシード別の曲線を全部描く（手法の数だけパネル）。"""
    n = len(mats)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (m, mat) in zip(axes, mats):
        for i, row in enumerate(mat):
            ax.plot(grid / 1e6, smooth(row, window), lw=1, alpha=0.55,
                    label=f"seed {i}")
        ax.plot(grid / 1e6, smooth(mat.mean(axis=0), window),
                color="black", lw=2, label="mean")
        ax.axhline(COLLAPSE, color="gray", ls="--", lw=1)
        ax.set_xlabel("Timesteps (M)")
        ax.set_title(m["label"])
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Episodic return (eval, mean of 5 episodes)")
    fig.suptitle("HalfCheetah-v5: per-seed learning curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_final_window(finals_by_method, out_path):
    """図3: シード別の最終区間平均（棒、手法ごとに並べる）+ 崩壊ライン + IQM。"""
    methods = [(m, f) for m, f in finals_by_method if len(f) == 5]
    if not methods:
        print("  [注意] 最終区間の図は 5 シード揃った手法が無いため作りません")
        return
    n_seeds = 5
    seeds = np.arange(n_seeds)
    n_m = len(methods)
    width = 0.8 / n_m
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for j, (m, finals) in enumerate(methods):
        offset = (j - (n_m - 1) / 2) * width
        ax.bar(seeds + offset, finals, width, color=m["color"],
               label=m["label"])
    ax.axhline(COLLAPSE, color="red", ls="--", lw=1.2,
               label="Collapse threshold (4,000)")
    # IQM は右端の余白に手法ごとのマーカーで示す
    for j, (m, finals) in enumerate(methods):
        ax.scatter([n_seeds + 0.1 + j * 0.28], [iqm(finals)], marker="D",
                   s=90, color=m["color"], edgecolor="black", zorder=5,
                   label=f"IQM ({m['key']})")
    ax.set_xlim(-0.7, n_seeds + 0.1 + n_m * 0.28 + 0.2)
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


def diag_panel(ax, series_list, total_steps, title, ylabel,
               logy=False, hline=None, hline_label=None):
    """診断パネル共通処理。series_list は (配列, 色, ラベル) のリスト。
    配列が複数シード分ある場合は細線＋平均の太線で描く想定で、
    呼び出し側でシード平均を作って渡してもよい。"""
    for y, color, label in series_list:
        b = bin_reduce(y)
        xs = diag_x(len(y), total_steps)[:len(b)] / 1e6
        ax.plot(xs, b, color=color, lw=1.4, label=label)
    if hline is not None:
        ax.axhline(hline, color="red", ls="--", lw=1.2,
                   label=hline_label or f"target ({hline:g})")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Training steps (M, approx. from log index)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)


def seed_mean_series(series):
    """シードごとの配列を区間圧縮してから平均した1本を返す。"""
    if not series:
        return None
    binned = [bin_reduce(y) for y in series]
    n = min(len(b) for b in binned)
    return np.array([b[:n] for b in binned]).mean(axis=0), len(series[0])


def plot_fixed_diagnostics(diag, total_steps, target_kl, out_path):
    """図4: 先輩版の診断 2x2（KL / β / エントロピー / α）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    color = "tab:green"

    def one_panel(ax, key, title, ylabel, logy=False, hline=None):
        got = seed_mean_series(diag.get(key, []))
        if got is None:
            ax.set_title(f"{title} (no data)")
            return
        mean_series, n_points = got
        # seed_mean_series は圧縮済みなので、横軸は圧縮点数基準で作り直す
        xs = np.linspace(0, total_steps, len(mean_series)) / 1e6
        ax.plot(xs, mean_series, color=color, lw=1.8,
                label="mean over seeds")
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

    one_panel(axes[0, 0], "kl_values", "KL divergence (senior ADR)",
              "KL (total)", hline=target_kl)
    one_panel(axes[0, 1], "beta_logs", "Beta (ADR coefficient)", "beta",
              logy=True)
    one_panel(axes[1, 0], "pi_entropies",
              "Policy entropy (pre-tanh Gaussian)", "Entropy")
    one_panel(axes[1, 1], "alpha_logs", "Alpha (entropy coefficient)",
              "alpha")
    fig.suptitle("Fixed prior (senior ADR): diagnostics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_kl_beta_comparison(d_diag, f_diag, total_steps,
                            d_target, f_target, out_path):
    """図5: ADR 2手法の KL・β 比較 2x2（目標値が違うのでパネルを分ける）。"""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def one_panel(ax, diag, key, color, title, ylabel, logy=False,
                  hline=None):
        got = seed_mean_series(diag.get(key, []))
        if got is None:
            ax.set_title(f"{title} (no data)")
            return
        mean_series, _ = got
        xs = np.linspace(0, total_steps, len(mean_series)) / 1e6
        ax.plot(xs, mean_series, color=color, lw=1.8,
                label="mean over seeds")
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

    one_panel(axes[0, 0], d_diag, "kl_values", "tab:blue",
              "KL: delayed anchor", "KL (total)", hline=d_target)
    one_panel(axes[0, 1], f_diag, "kl_values", "tab:green",
              "KL: fixed prior (senior ADR)", "KL (total)", hline=f_target)
    one_panel(axes[1, 0], d_diag, "beta_logs", "tab:blue",
              "Beta: delayed anchor", "beta", logy=True)
    one_panel(axes[1, 1], f_diag, "beta_logs", "tab:green",
              "Beta: fixed prior (senior ADR)", "beta", logy=True)
    fig.suptitle("ADR diagnostics: delayed anchor vs fixed prior")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def plot_entropy_alpha(diags_by_method, total_steps, out_path):
    """図6: 3手法のエントロピーと α の重ね図。diags_by_method は
    (method, diag) のリスト。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title, ylabel in (
            (axes[0], "pi_entropies", "Policy entropy (pre-tanh Gaussian)",
             "Entropy"),
            (axes[1], "alpha_logs", "Alpha (entropy coefficient)",
             "alpha")):
        for m, diag in diags_by_method:
            got = seed_mean_series(diag.get(key, []))
            if got is None:
                continue
            mean_series, _ = got
            xs = np.linspace(0, total_steps, len(mean_series)) / 1e6
            ax.plot(xs, mean_series, color=m["color"], lw=2,
                    label=m["label"])
        ax.set_xlabel("Training steps (M, approx. from log index)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("All methods: entropy and alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [保存] {out_path}")


def main():
    ap = argparse.ArgumentParser(description="HalfCheetah 3手法比較のプロット")
    ap.add_argument("--delayed-dir",
                    default="./npz_logs_delayed_anchor_halfcheetah")
    ap.add_argument("--fixed-dir",
                    default="./npz_logs_fixed_prior_halfcheetah")
    ap.add_argument("--sac-dir", default="./npz_logs_pure_sac_halfcheetah")
    ap.add_argument("--out-dir", default="./plots_three_way")
    ap.add_argument("--smooth", type=int, default=11,
                    help="表示用の移動平均の幅（評価点の個数、既定 11 = 55k steps）")
    ap.add_argument("--delayed-target-kl", type=float, default=0.30,
                    help="遅延アンカーの KL 目標（合計）。既定 0.30")
    ap.add_argument("--fixed-target-kl", type=float, default=6.0,
                    help="先輩版の KL 目標（合計）。既定 6.0 = 1.0/次元 × 6次元")
    args = ap.parse_args()
    dir_by_arg = {"delayed_dir": args.delayed_dir,
                  "fixed_dir": args.fixed_dir,
                  "sac_dir": args.sac_dir}
    target_by_key = {"delayed": args.delayed_target_kl,
                     "fixed": args.fixed_target_kl}

    seeds = range(5)
    curves_by_key = {}
    for m in METHODS:
        print(f"== {m['short']} ==")
        curves_by_key[m["key"]] = load_curves(
            dir_by_arg[m["dir_arg"]], m["pattern"], seeds)
    present = [m for m in METHODS if curves_by_key[m["key"]]]
    if not present:
        print("どの手法の npz も読めませんでした。終了します。")
        return

    finals_by_method = []
    for m in present:
        finals, _ = summarize(m["short"], curves_by_key[m["key"]])
        finals_by_method.append((m, finals))

    # 共通グリッドは最初に読めた曲線の timesteps を使う（評価周期は同一のはず）
    first_curves = curves_by_key[present[0]["key"]]
    grid = first_curves[0][0]
    total_steps = float(grid.max())
    mats = [(m, interp_all(curves_by_key[m["key"]], grid))
            for m in present]

    os.makedirs(args.out_dir, exist_ok=True)
    print("\n== 図の作成 ==")
    plot_overlay(grid, mats, args.smooth,
                 os.path.join(args.out_dir, "learning_curves_overlay.png"))
    plot_by_seed(grid, mats, args.smooth,
                 os.path.join(args.out_dir, "learning_curves_by_seed.png"))
    plot_final_window(finals_by_method,
                      os.path.join(args.out_dir, "final_window.png"))

    # 診断
    print("\n== 診断ログの読み込み ==")
    diag_keys = ["kl_values", "beta_logs", "pi_entropies", "alpha_logs"]
    diags = {}
    for m in present:
        # SAC は KL・β を記録していないので、その分は最初から読まない
        keys = diag_keys if m["key"] != "sac" else ["pi_entropies",
                                                    "alpha_logs"]
        diags[m["key"]] = load_diag(dir_by_arg[m["dir_arg"]], m["pattern"],
                                    seeds, keys)
        summarize_diag(m["short"], diags[m["key"]])

    if diags.get("fixed", {}).get("kl_values"):
        plot_fixed_diagnostics(
            diags["fixed"], total_steps, target_by_key["fixed"],
            os.path.join(args.out_dir, "diagnostics_fixed_prior.png"))
    else:
        print("  [注意] 先輩版に kl_values が無いため図4は作りません")
    if diags.get("delayed", {}).get("kl_values") and \
            diags.get("fixed", {}).get("kl_values"):
        plot_kl_beta_comparison(
            diags["delayed"], diags["fixed"], total_steps,
            target_by_key["delayed"], target_by_key["fixed"],
            os.path.join(args.out_dir, "diagnostics_kl_beta_comparison.png"))
    else:
        print("  [注意] KL が揃わないため図5は作りません")
    if any(diags[m["key"]].get("pi_entropies") for m in present):
        plot_entropy_alpha(
            [(m, diags[m["key"]]) for m in present], total_steps,
            os.path.join(args.out_dir, "diagnostics_entropy_alpha.png"))
    else:
        print("  [注意] エントロピーが無いため図6は作りません")


if __name__ == "__main__":
    main()
