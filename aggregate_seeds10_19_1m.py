#!/usr/bin/env python3
"""seed 10〜19 傾向確認ラン集計スクリプト (SAC vs 先輩版 ADR / HalfCheetah-v5、1M steps)

keisanB のリポジトリ直下で実行する想定:
    python3 aggregate_seeds10_19_1m.py

読み込む npz (シード別):
    npz_logs_pure_sac_halfcheetah/pure_sac_halfcheetah_seed{10..19}.npz
    npz_logs_fixed_prior_halfcheetah/fixed_prior_halfcheetah_seed{10..19}.npz
各 npz に timesteps / returns / pi_entropies / alpha_logs
(先輩版は kl_values / beta_logs も) が入っている前提。無いファイルは
警告して読み飛ばす。

初期分布の記録 (あれば) も読む:
    npz_logs_initdist_halfcheetah/{pure_sac|fixed_prior}_halfcheetah_seed{10..19}_initdist.npz
    (sigma_mean_per_dim / mu_mean_per_dim)

出力 (./plots_seeds10_19_1m/ 以下):
    summary_per_seed.csv   シード別の最終区間平均・AUC・崩壊／保留フラグ
    distribution_per_seed.csv シード別のエントロピー・α (先輩版は KL・β も) の区間平均
    fig_curves_by_seed.png 手法ごとに10シード全曲線 + 平均
    fig_mean_curves.png    両手法の平均曲線 (mean ± std の帯)
    fig_final_window.png   シード別 最終区間平均の棒グラフ
    fig_distribution_curves.png エントロピーと α の全シード曲線 (手法ごと)
標準出力にも集計表を出すので、結果は貼り付けでも共有できる。

このランは 3M の 10 シード集計とは別枠の「傾向確認」である。
集計値は 3M の集計に混ぜないこと。問うのは走らせる前に固定した
2 点だけ (run_halfcheetah_seeds10_19_1m.py の冒頭を参照)。
  (1) 這い型の崩壊は新しい seed でも起きるか
  (2) α の序盤の分離 (100〜500k) は未見の seed でも当たるか

判定の線引き (走らせる前に固定。結果を見てから動かさない):
    最終区間 = 500k〜1M steps の評価点
    崩壊     = 最終区間平均 < 4,000
    判定保留 = 4,000 <= 最終区間平均 < 7,000 (無理に二分しない)
    健全     = 最終区間平均 >= 7,000
    IQM      = rliable 方式 (中央50%の重み付き平均)
    CI       = ブートストラップ (10,000回) の百分位法
    AUC      = 評価系列の台形積分 (NumPy 2.x の trapezoid / 1.x の trapz 両対応)
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = list(range(10, 20))
TOTAL_STEPS = 1_000_000
FINAL_FROM = 500_000
COLLAPSE_LINE = 4_000.0
HEALTHY_LINE = 7_000.0
OUTDIR = "./plots_seeds10_19_1m"

ARMS = {
    "SAC": "./npz_logs_pure_sac_halfcheetah/pure_sac_halfcheetah_seed{}.npz",
    "ADR": "./npz_logs_fixed_prior_halfcheetah/fixed_prior_halfcheetah_seed{}.npz",
}
INITDIST = {
    "SAC": "./npz_logs_initdist_halfcheetah/pure_sac_halfcheetah_seed{}_initdist.npz",
    "ADR": "./npz_logs_initdist_halfcheetah/fixed_prior_halfcheetah_seed{}_initdist.npz",
}

_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def iqm(x):
    xs = np.sort(np.asarray(x, dtype=float))
    n = len(xs)
    w = []
    for i in range(n):
        lo, hi = i / n, (i + 1) / n
        w.append(max(0.0, min(hi, 0.75) - max(lo, 0.25)))
    w = np.array(w)
    return float((w * xs).sum() / w.sum())


def boot_ci(x, fn, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    vals = [fn(rng.choice(x, size=len(x), replace=True)) for _ in range(reps)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_arm(pattern):
    """{seed: (timesteps, returns)} を返す。読めないファイルは警告して飛ばす。"""
    out = {}
    for s in SEEDS:
        path = pattern.format(s)
        if not os.path.exists(path):
            print(f"[warn] not found: {path}")
            continue
        try:
            d = np.load(path)
            ts, rw = np.asarray(d["timesteps"]), np.asarray(d["returns"])
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to load {path}: {e}")
            continue
        out[s] = (ts, rw)
    return out


WINDOWS = [
    ("0-100k", 0, 100_000),
    ("100-500k", 100_000, 500_000),
    ("500k-1M", 500_000, 1_000_000),
]
DIST_KEYS = ["pi_entropies", "alpha_logs", "kl_values", "beta_logs"]


def window_slice(n, lo, hi, total_steps=TOTAL_STEPS):
    """diagnose_collapse_seed.py と同じ割り振り: 系列の位置を steps に線形換算。"""
    i0 = int(round(lo / total_steps * n))
    i1 = int(round(hi / total_steps * n))
    return slice(max(0, i0), min(n, i1))


def window_means(arr):
    if arr is None or len(arr) == 0:
        return [float("nan")] * len(WINDOWS)
    return [float(np.nanmean(arr[window_slice(len(arr), lo, hi)]))
            for _, lo, hi in WINDOWS]


def load_series(pattern):
    """{seed: {key: array}} を返す (分布側の系列。無いキーは欠けたまま)。"""
    out = {}
    for s in SEEDS:
        path = pattern.format(s)
        if not os.path.exists(path):
            continue
        try:
            d = np.load(path)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to load {path}: {e}")
            continue
        out[s] = {k: np.asarray(d[k], dtype=float) for k in DIST_KEYS if k in d.files}
    return out


def load_initdist(pattern):
    """{seed: sigma_mean_per_dim} を返す。無いファイルは飛ばす。"""
    out = {}
    for s in SEEDS:
        path = pattern.format(s)
        if not os.path.exists(path):
            continue
        try:
            d = np.load(path)
            out[s] = np.asarray(d["sigma_mean_per_dim"], dtype=float)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] failed to load {path}: {e}")
            continue
    return out


def downsample(arr, blocks=300):
    if arr is None or len(arr) == 0:
        return np.full(blocks, np.nan)
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= blocks:
        # 系列が短いときは線形補間で blocks 点にそろえる (空チャンクを作らない)
        x = np.linspace(0, len(arr) - 1, blocks)
        return np.interp(x, np.arange(len(arr)), arr)
    chunks = np.array_split(arr, blocks)
    return np.array([float(np.nanmean(c)) if len(c) else np.nan for c in chunks])


def classify(final_mean):
    if final_mean < COLLAPSE_LINE:
        return "collapsed"
    if final_mean < HEALTHY_LINE:
        return "hold"
    return "healthy"


def seed_stats(ts, rw):
    fw = rw[ts >= FINAL_FROM]
    m500 = ts <= 500_000
    fm = float(fw.mean())
    return dict(
        final_mean=fm,
        final_min=float(fw.min()),
        final_max=float(fw.max()),
        final_std=float(fw.std()),
        auc_500k=float(_trapz(rw[m500], ts[m500])),
        auc_1m=float(_trapz(rw, ts)),
        status=classify(fm),
    )


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = {name: load_arm(pat) for name, pat in ARMS.items()}
    stats = {name: {s: seed_stats(*data[name][s]) for s in data[name]} for name in ARMS}

    # ---- CSV ----
    csv_path = os.path.join(OUTDIR, "summary_per_seed.csv")
    with open(csv_path, "w") as f:
        f.write("algo,seed,final_mean,final_min,final_max,final_std,auc_0_500k,auc_0_1m,status\n")
        for name in ARMS:
            for s in sorted(stats[name]):
                st = stats[name][s]
                f.write(f"{name},{s},{st['final_mean']:.2f},{st['final_min']:.2f},"
                        f"{st['final_max']:.2f},{st['final_std']:.2f},"
                        f"{st['auc_500k']:.6e},{st['auc_1m']:.6e},{st['status']}\n")

    # ---- 集計表 (標準出力) ----
    finals = {name: np.array([stats[name][s]["final_mean"] for s in sorted(stats[name])])
              for name in ARMS}
    print("\n=== Final-window (500k-1M) per seed ===")
    print(f"{'seed':>4} | {'SAC':>10} | {'ADR':>10} | winner | status (SAC/ADR)")
    sac_seeds = sorted(stats["SAC"])
    adr_seeds = sorted(stats["ADR"])
    for s in sorted(set(sac_seeds) & set(adr_seeds)):
        a, b = stats["SAC"][s]["final_mean"], stats["ADR"][s]["final_mean"]
        sa, sb = stats["SAC"][s]["status"], stats["ADR"][s]["status"]
        print(f"{s:>4} | {a:>10.1f} | {b:>10.1f} | {'SAC' if a > b else 'ADR'} | {sa}/{sb}")
    print("\n=== Aggregates (1M runs only; do NOT mix into the 3M 10-seed aggregates) ===")
    for name in ARMS:
        x = finals[name]
        ilo, ihi = boot_ci(x, iqm)
        mlo, mhi = boot_ci(x, lambda v: float(np.mean(v)))
        auc500 = np.array([stats[name][s]["auc_500k"] for s in sorted(stats[name])])
        auc1m = np.array([stats[name][s]["auc_1m"] for s in sorted(stats[name])])
        n_col = sum(1 for s in sorted(stats[name]) if stats[name][s]["status"] == "collapsed")
        n_hold = sum(1 for s in sorted(stats[name]) if stats[name][s]["status"] == "hold")
        print(f"{name}: n={len(x)}")
        print(f"  mean   = {x.mean():,.1f} (95%CI {mlo:,.0f} - {mhi:,.0f})")
        print(f"  median = {np.median(x):,.1f}")
        print(f"  IQM    = {iqm(x):,.1f} (95%CI {ilo:,.0f} - {ihi:,.0f})")
        print(f"  collapsed = {n_col}/{len(x)} | hold = {n_hold}/{len(x)}")
        print(f"  healthy-only mean = {x[x >= HEALTHY_LINE].mean() if (x >= HEALTHY_LINE).any() else float('nan'):,.1f}")
        print(f"  AUC(0-500k) mean = {auc500.mean():.4e} | AUC(0-1M) mean = {auc1m.mean():.4e}")

    # ---- 図1: 手法ごとに全シード曲線 + 平均 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, (name, color) in zip(axes, [("SAC", "#4C72B0"), ("ADR", "#DD8452")]):
        series = [data[name][s] for s in sorted(data[name])]
        maxlen = max(len(rw) for _, rw in series)
        mat = np.full((len(series), maxlen), np.nan)
        ts0 = series[0][0]
        for i, (_, rw) in enumerate(series):
            mat[i, :len(rw)] = rw
        for i, (ts, rw) in enumerate(series):
            ax.plot(ts / 1e6, rw, lw=0.7, alpha=0.45, color=color)
        with np.errstate(all="ignore"):
            mean_curve = np.nanmean(mat, axis=0)
        ax.plot(ts0 / 1e6, mean_curve, lw=2.2, color=color, label="mean")
        ax.axhline(COLLAPSE_LINE, color="crimson", ls="--", lw=1.0)
        ax.set_xlabel("Steps (M)")
        ax.set_title(f"{name} (seeds 10-19, 1M)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Eval return")
    fig.suptitle("HalfCheetah-v5 learning curves by seed (1M runs)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_curves_by_seed.png"), dpi=200)
    plt.close(fig)

    # ---- 図2: 両手法の平均曲線 (mean ± std) ----
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for name, color in [("SAC", "#4C72B0"), ("ADR", "#DD8452")]:
        series = [data[name][s] for s in sorted(data[name])]
        maxlen = max(len(rw) for _, rw in series)
        mat = np.full((len(series), maxlen), np.nan)
        ts0 = series[0][0]
        for i, (_, rw) in enumerate(series):
            mat[i, :len(rw)] = rw
        with np.errstate(all="ignore"):
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
        ax.plot(ts0 / 1e6, mu, lw=2.0, color=color, label=name)
        ax.fill_between(ts0 / 1e6, mu - sd, mu + sd, color=color, alpha=0.18)
    ax.axhline(COLLAPSE_LINE, color="crimson", ls="--", lw=1.0, label="collapse threshold (4,000)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("Eval return (mean ± std over 10 seeds)")
    ax.set_title("HalfCheetah-v5: SAC vs ADR (FixedPrior), seeds 10-19 (1M)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_mean_curves.png"), dpi=200)
    plt.close(fig)

    # ---- 図3: シード別 最終区間平均 ----
    fig, ax = plt.subplots(figsize=(9, 4.6))
    xs = np.arange(len(SEEDS))
    wdt = 0.38
    sac_vals = [stats["SAC"][s]["final_mean"] if s in stats["SAC"] else np.nan for s in SEEDS]
    adr_vals = [stats["ADR"][s]["final_mean"] if s in stats["ADR"] else np.nan for s in SEEDS]
    ax.bar(xs - wdt / 2, sac_vals, wdt, label="SAC", color="#4C72B0")
    ax.bar(xs + wdt / 2, adr_vals, wdt, label="ADR (FixedPrior)", color="#DD8452")
    ax.axhline(COLLAPSE_LINE, color="crimson", ls="--", lw=1.2, label="collapse threshold (4,000)")
    for x, v in zip(xs - wdt / 2, sac_vals):
        if np.isfinite(v):
            ax.annotate(f"{v:,.0f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    for x, v in zip(xs + wdt / 2, adr_vals):
        if np.isfinite(v):
            ax.annotate(f"{v:,.0f}", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"seed {i}" for i in SEEDS])
    ax.set_ylabel("Final-window mean return (500k-1M)")
    ax.set_title("HalfCheetah-v5: SAC vs ADR (FixedPrior), seeds 10-19 (1M)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 16000)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_final_window.png"), dpi=200)
    plt.close(fig)

    # ---- 分布側の集計: エントロピー・α (先輩版は KL・β も) ----
    series = {name: load_series(pat) for name, pat in ARMS.items()}
    dist = {}  # dist[algo][seed][key] = 区間平均のリスト
    for name in ARMS:
        dist[name] = {}
        for s, dd in series[name].items():
            dist[name][s] = {k: window_means(dd.get(k)) for k in DIST_KEYS}

    dist_csv = os.path.join(OUTDIR, "distribution_per_seed.csv")
    win_names = [w[0] for w in WINDOWS]
    with open(dist_csv, "w") as f:
        cols = ["algo", "seed", "status"]
        for key, short in [("pi_entropies", "ent"), ("alpha_logs", "alpha"),
                           ("kl_values", "kl"), ("beta_logs", "beta")]:
            cols += [f"{short}_{w}" for w in win_names]
        f.write(",".join(cols) + "\n")
        for name in ARMS:
            for s in sorted(dist[name]):
                row = [name, str(s), stats[name][s]["status"]]
                for key in DIST_KEYS:
                    row += [f"{v:.4f}" for v in dist[name][s][key]]
                f.write(",".join(row) + "\n")

    # ---- 問い (2) の記述的確認: 100-500k の α (先輩版は β も) ----
    print("\n=== Alpha separation check (100-500k window, pre-fixed question 2) ===")
    print("No threshold is fitted here: only whether the groups separate without")
    print("overlap, and whether the direction matches (healthy rising, collapsed falling).")
    for name in ARMS:
        for group, sel in [("collapsed", "collapsed"), ("healthy", "healthy"), ("hold", "hold")]:
            seeds_g = [s for s in sorted(dist[name])
                       if stats[name][s]["status"] == sel]
            if not seeds_g:
                continue
            alp = np.array([dist[name][s]["alpha_logs"][1] for s in seeds_g])
            ent = np.array([dist[name][s]["pi_entropies"][1] for s in seeds_g])
            line = (f"{name} {group:>9} (n={len(seeds_g)}): "
                    f"alpha(100-500k) = {np.nanmean(alp):.4f} "
                    f"(min {np.nanmin(alp):.4f}, max {np.nanmax(alp):.4f}), "
                    f"entropy = {np.nanmean(ent):.2f}")
            if name == "ADR":
                be = np.array([dist[name][s]["beta_logs"][1] for s in seeds_g])
                line += (f", beta = {np.nanmean(be):.4f} "
                         f"(min {np.nanmin(be):.4f}, max {np.nanmax(be):.4f})")
            print(line)

    print("\n=== Distribution (final window 500k-1M means) ===")
    for name in ARMS:
        for group, sel in [("collapsed", "collapsed"), ("healthy", "healthy"), ("hold", "hold")]:
            seeds_g = [s for s in sorted(dist[name])
                       if stats[name][s]["status"] == sel]
            if not seeds_g:
                continue
            ent = np.array([dist[name][s]["pi_entropies"][-1] for s in seeds_g])
            alp = np.array([dist[name][s]["alpha_logs"][-1] for s in seeds_g])
            line = (f"{name} {group:>9} (n={len(seeds_g)}): "
                    f"entropy = {np.nanmean(ent):.2f} ± {np.nanstd(ent):.2f}, "
                    f"alpha = {np.nanmean(alp):.4f} ± {np.nanstd(alp):.4f}")
            if name == "ADR":
                kl = np.array([dist[name][s]["kl_values"][-1] for s in seeds_g])
                be = np.array([dist[name][s]["beta_logs"][-1] for s in seeds_g])
                line += (f", KL = {np.nanmean(kl):.3f} ± {np.nanstd(kl):.3f}, "
                         f"beta = {np.nanmean(be):.4f} ± {np.nanstd(be):.4f}")
            print(line)
    print("(per-seed window means: distribution_per_seed.csv)")

    # ---- 図4: エントロピーと α の全シード曲線 ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    xgrid = np.linspace(0, 1.0, 300)
    for col, (name, color) in enumerate([("SAC", "#4C72B0"), ("ADR", "#DD8452")]):
        for row, (key, ylabel) in enumerate(
                [("pi_entropies", "Policy entropy"), ("alpha_logs", "alpha")]):
            ax = axes[row][col]
            healthy_curves, collapsed_curves, hold_curves = [], [], []
            for s in sorted(series[name]):
                if key not in series[name][s]:
                    continue
                curve = downsample(series[name][s][key])
                st = stats[name][s]["status"]
                c = "crimson" if st == "collapsed" else ("orange" if st == "hold" else color)
                ax.plot(xgrid, curve, lw=0.8, color=c,
                        alpha=0.35 if st == "collapsed" else 0.55)
                (collapsed_curves if st == "collapsed"
                 else hold_curves if st == "hold" else healthy_curves).append(curve)
            for curves, lab, c in [(healthy_curves, "healthy mean", color),
                                   (hold_curves, "hold mean", "orange"),
                                   (collapsed_curves, "collapsed mean", "crimson")]:
                if curves:
                    ax.plot(xgrid, np.nanmean(np.array(curves), axis=0),
                            lw=2.2, color=c, label=lab)
            ax.set_title(f"{name}: {ylabel}" if row == 0 else f"{name}")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
            if row == 1:
                ax.set_xlabel("Steps (M)")
    fig.suptitle("Entropy and alpha trajectories by seed (red = collapsed, orange = hold)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig_distribution_curves.png"), dpi=200)
    plt.close(fig)

    # ---- 初期分布の要約 (あれば) ----
    init = {name: load_initdist(pat) for name, pat in INITDIST.items()}
    if init["SAC"] or init["ADR"]:
        print("\n=== Initial policy sigma (mean over dims, per seed) ===")
        for s in SEEDS:
            sa = init["SAC"].get(s)
            sb = init["ADR"].get(s)
            parts = []
            if sa is not None:
                parts.append(f"SAC sigma-mean {sa.mean():.3f}")
            if sb is not None:
                parts.append(f"ADR sigma-mean {sb.mean():.3f}")
            if sa is not None and sb is not None:
                parts.append(f"max |diff| per dim {np.max(np.abs(sa - sb)):.4f}")
            if parts:
                print(f"seed {s}: " + " | ".join(parts))

    print(f"\nSaved: {csv_path}, {dist_csv} and 4 figures in {OUTDIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
