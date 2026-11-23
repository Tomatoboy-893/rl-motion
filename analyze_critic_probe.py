#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_critic_probe.py — ステップ2（クリティック記録つき再走）の npz 読み取り

run_critic_probe_halfcheetah.py が出した *_criticprobe.npz を読み、
ランごとに以下を出す。複数ファイルを渡すと最後に横並びの要約表も出す。

1. 再現チェック: 評価リターンの 500k 時点・最終区間（900k〜1M）平均・最大値。
   元ランの seed 8 は 500k で約2,100〜2,200（這い型）、seed 7 は約5,700〜5,900。
2. 区間平均（0–100k / 100–500k / 500k–1M）:
   リターン、エントロピー、α、（先輩版は KL・β）、プローブ記録
   （Q1/Q2 平均、目標Q、Q のばらつき、|TD誤差|、critic loss、勾配ノルム、
   休眠率、next_log_prob 平均）。
3. α の問い: α の更新が使う next_log_prob（の符号反転）と、記録上の
   エントロピーを並べ、序盤でどちらがどう動いたかを見る。
4. 時点スナップショット（100k / 250k / 500k / 750k / 1M 付近）:
   リターンと α。前後関係の当たりを付けるため。

読み分けの目安（計画文書フェーズ3ステップ2）:
- Q が膨らみ続ける → 過大評価・発散型（Hussing et al. 2024）
- Q は安定だが TD 誤差が下がらず張り付く → primacy 型（Nikishin et al. 2022）
- Q が健全側と同型で、這い歩行を妥当に評価 → 局所最適型
判定は必ず同手法の健全側（seed 7）と並べて行うこと。単独では判定しない。

使い方:
  python3 analyze_critic_probe.py <npz> [<npz> ...] [--plot] [--out DIR]
  --plot を付けると、ランごとの図（8パネル）と、同じ手法の2ランを
  重ねた比較図（seed 8 対 seed 7）を PNG で保存する（既定 ./plots_critic_probe/）。
"""

import sys
import numpy as np

WINDOWS = [(0, 100_000), (100_000, 500_000), (500_000, 1_000_000)]
SNAP_STEPS = [100_000, 250_000, 500_000, 750_000, 1_000_000]

PROBE_KEYS = [
    "q1_mean", "q2_mean", "target_q_mean", "q_std",
    "td_abs_mean", "td_std", "critic_loss_probe",
    "critic_grad_norm", "actor_grad_norm",
    "dormant_frac_qf0", "dormant_frac_qf1", "next_log_prob_mean",
]


def wmean_by_steps(steps, values, lo, hi):
    """steps が [lo, hi) に入る values の平均（NaN 無視）。無ければ NaN。"""
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    m = (steps >= lo) & (steps < hi)
    if not m.any() or np.all(np.isnan(values[m])):
        return float("nan")
    return float(np.nanmean(values[m]))


def wmean_positional(values, total_steps, lo, hi):
    """timesteps を持たないログ配列を、全体に対する位置の割合で区間平均する。
    diagnose_collapse_seed.py と同じ位置スライスの流儀。"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return float("nan")
    i0 = int(n * lo / total_steps)
    i1 = max(int(n * hi / total_steps), i0 + 1)
    return float(np.nanmean(values[i0:i1]))


def value_near_step(steps, values, target):
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(steps) == 0:
        return float("nan")
    idx = int(np.argmin(np.abs(steps - target)))
    return float(values[idx])


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:.{nd}f}"


def analyze(path):
    d = np.load(path)
    keys = set(d.files)
    t = d["timesteps"]
    r = d["returns"]
    total = int(t.max()) if len(t) else 0
    pt = d["probe_timesteps"] if "probe_timesteps" in keys else np.array([])

    print("=" * 78)
    print(path)
    print(f"  eval points: {len(t)}  (max step {total:,}), "
          f"probe points: {len(pt)}")
    print("-" * 78)

    # 1. 再現チェック
    r500 = value_near_step(t, r, 500_000)
    tail = r[t >= 900_000] if (t >= 900_000).any() else r[-10:]
    print(f"  [再現] return@500k = {fmt(r500, 1)}   "
          f"最終区間(900k-)平均 = {fmt(float(np.mean(tail)), 1)}   "
          f"最大 = {fmt(float(np.max(r)), 1)}")

    # 2. 区間平均: リターン（timesteps 基準）
    print("  [区間平均]        0-100k    100-500k   500k-1M")
    rows = [("returns", [wmean_by_steps(t, r, lo, hi) for lo, hi in WINDOWS], 1)]
    if "pi_entropies" in keys:
        rows.append(("entropy", [wmean_positional(d["pi_entropies"], total, lo, hi)
                                 for lo, hi in WINDOWS], 3))
    if "alpha_logs" in keys:
        rows.append(("alpha", [wmean_positional(d["alpha_logs"], total, lo, hi)
                               for lo, hi in WINDOWS], 4))
    if "kl_values" in keys:
        rows.append(("KL", [wmean_positional(d["kl_values"], total, lo, hi)
                            for lo, hi in WINDOWS], 3))
    if "beta_logs" in keys:
        rows.append(("beta", [wmean_positional(d["beta_logs"], total, lo, hi)
                              for lo, hi in WINDOWS], 4))
    for name, vals, nd in rows:
        print(f"    {name:<16}" + "".join(f"{fmt(v, nd):>10}" for v in vals))

    # プローブ記録（probe_timesteps 基準）
    for k in PROBE_KEYS:
        if k in keys:
            vals = [wmean_by_steps(pt, d[k], lo, hi) for lo, hi in WINDOWS]
            nd = 4 if "frac" in k or "grad" in k else 3
            print(f"    {k:<16}" + "".join(f"{fmt(v, nd):>10}" for v in vals))

    # 3. α の問い: -next_log_prob と entropy の並び（区間平均）
    if "next_log_prob_mean" in keys and "pi_entropies" in keys:
        nlp = [-wmean_by_steps(pt, d["next_log_prob_mean"], lo, hi)
               for lo, hi in WINDOWS]
        ent = [wmean_positional(d["pi_entropies"], total, lo, hi)
               for lo, hi in WINDOWS]
        print("  [αの問い] -next_log_prob（α調整が見ている広がりの目安） vs 記録エントロピー")
        print(f"    {'-next_log_prob':<16}" + "".join(f"{fmt(v):>10}" for v in nlp))
        print(f"    {'entropy(記録)':<16}" + "".join(f"{fmt(v):>10}" for v in ent))

    # 4. 時点スナップショット
    print("  [時点] step:      " + "".join(f"{s // 1000:>9}k" for s in SNAP_STEPS))
    print(f"    {'returns':<16}" + "".join(
        f"{fmt(value_near_step(t, r, s), 0):>10}" for s in SNAP_STEPS))
    if "alpha_logs" in keys:
        a = d["alpha_logs"]
        n = len(a)
        avals = [float(a[min(int(n * s / total), n - 1)]) if n else float("nan")
                 for s in SNAP_STEPS]
        print(f"    {'alpha':<16}" + "".join(f"{fmt(v, 4):>10}" for v in avals))
    print()
    return {
        "path": path,
        "r500": r500,
        "tail": float(np.mean(tail)),
        "q_tail": wmean_by_steps(pt, d["q1_mean"], 500_000, 1_000_000)
        if "q1_mean" in keys else float("nan"),
        "td_tail": wmean_by_steps(pt, d["td_abs_mean"], 500_000, 1_000_000)
        if "td_abs_mean" in keys else float("nan"),
    }


def positional_x(values, total):
    """timesteps を持たないログ配列の x 軸（全体への位置で近似）。"""
    return np.linspace(0, total, len(values))


def plot_run(path, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = np.load(path)
    keys = set(d.files)
    t = d["timesteps"]
    total = int(t.max()) if len(t) else 0
    pt = d["probe_timesteps"] if "probe_timesteps" in keys else np.array([])
    name = path.split("/")[-1].replace(".npz", "")

    fig, axes = plt.subplots(4, 2, figsize=(13, 14))
    fig.suptitle(name)
    ax = axes.ravel()

    ax[0].plot(t, d["returns"], lw=0.8)
    ax[0].set_title("eval returns")
    ax[0].set_xlabel("steps")

    for k, lab in [("q1_mean", "Q1"), ("q2_mean", "Q2"),
                   ("target_q_mean", "target Q")]:
        if k in keys:
            ax[1].plot(pt, d[k], lw=0.8, label=lab)
    ax[1].set_title("Q means (probe batch)")
    ax[1].legend(fontsize=8)

    if "td_abs_mean" in keys:
        ax[2].plot(pt, d["td_abs_mean"], lw=0.8, label="|TD| mean")
    if "td_std" in keys:
        ax[2].plot(pt, d["td_std"], lw=0.8, alpha=0.6, label="TD std")
    ax[2].set_title("TD error")
    ax[2].legend(fontsize=8)

    if "critic_loss_probe" in keys:
        ax[3].plot(pt, d["critic_loss_probe"], lw=0.8)
    ax[3].set_title("critic loss (probe)")

    if "pi_entropies" in keys:
        ax[4].plot(positional_x(d["pi_entropies"], total), d["pi_entropies"],
                   lw=0.8, label="entropy (logged)")
    if "next_log_prob_mean" in keys:
        ax[4].plot(pt, -d["next_log_prob_mean"], lw=0.8,
                   label="-next_log_prob (probe)")
    ax[4].set_title("entropy vs -next_log_prob")
    ax[4].legend(fontsize=8)

    if "alpha_logs" in keys:
        ax[5].plot(positional_x(d["alpha_logs"], total), d["alpha_logs"],
                   lw=0.8, label="alpha")
    if "beta_logs" in keys:
        ax[5].plot(positional_x(d["beta_logs"], total), d["beta_logs"],
                   lw=0.8, label="beta")
    ax[5].set_title("alpha / beta")
    ax[5].legend(fontsize=8)

    for k, lab in [("critic_grad_norm", "critic"), ("actor_grad_norm", "actor")]:
        if k in keys:
            ax[6].plot(pt, d[k], lw=0.8, label=lab)
    ax[6].set_title("grad norms")
    ax[6].legend(fontsize=8)

    plotted = False
    for k in ["dormant_frac_qf0", "dormant_frac_qf1"]:
        if k in keys and not np.all(np.isnan(d[k])):
            ax[7].plot(pt, d[k], lw=0.8, label=k)
            plotted = True
    ax[7].set_title("dormant fractions" if plotted else "dormant fractions (no data)")
    if plotted:
        ax[7].legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = f"{outdir}/plot_{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def plot_compare(paths, outdir):
    """同じ手法の2ラン（seed 8 崩壊・seed 7 健全想定）を重ねる。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for p in paths:
        d = np.load(p)
        label = p.split("/")[-1].replace("_criticprobe.npz", "")
        runs.append((label, d))
    algo = paths[0].split("/")[-1].split("_halfcheetah")[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"compare: {algo} (seed 8 vs seed 7)")
    ax = axes.ravel()
    colors = ["crimson", "steelblue"]

    def series(d, key, probe=False):
        keys = set(d.files)
        if key not in keys:
            return None, None
        if probe:
            return d["probe_timesteps"], d[key]
        if key in ("returns",):
            return d["timesteps"], d[key]
        total = int(d["timesteps"].max())
        return positional_x(d[key], total), d[key]

    panels = [
        ("returns", "returns", False),
        ("q1_mean", "Q1 mean", True),
        ("td_abs_mean", "|TD| mean", True),
        ("pi_entropies", "entropy", False),
        ("alpha_logs", "alpha", False),
        ("next_log_prob_mean", "-next_log_prob", True),
    ]
    for axi, (key, title, probe) in zip(ax, panels):
        for (label, d), c in zip(runs, colors):
            x, y = series(d, key, probe)
            if y is None:
                continue
            if key == "next_log_prob_mean":
                y = -y
            axi.plot(x, y, lw=0.8, color=c, label=label)
        axi.set_title(title)
        axi.grid(alpha=0.3)
        axi.legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f"{outdir}/compare_{algo}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  saved {out}")


def main():
    raw = sys.argv[1:]
    outdir = "plots_critic_probe"
    if "--out" in raw:
        i = raw.index("--out")
        outdir = raw[i + 1]
        raw = raw[:i] + raw[i + 2:]
    args = [a for a in raw if not a.startswith("--")]
    do_plot = "--plot" in raw
    if not args:
        print(__doc__)
        sys.exit(1)
    summaries = [analyze(p) for p in args]
    if do_plot:
        import os
        os.makedirs(outdir, exist_ok=True)
        for p in args:
            plot_run(p, outdir)
        from collections import defaultdict
        by_algo = defaultdict(list)
        for p in args:
            algo = p.split("/")[-1].split("_halfcheetah")[0]
            by_algo[algo].append(p)
        for algo, ps in by_algo.items():
            if len(ps) == 2:
                plot_compare(sorted(ps), outdir)
    if len(summaries) > 1:
        print("=" * 78)
        print("横並び要約（再現チェックと終盤のプローブ値）")
        print(f"  {'file':<52}{'r@500k':>9}{'最終平均':>10}{'Q1終盤':>9}{'|TD|終盤':>10}")
        for s in summaries:
            name = s["path"].split("/")[-1]
            print(f"  {name:<52}{fmt(s['r500'], 0):>9}{fmt(s['tail'], 0):>10}"
                  f"{fmt(s['q_tail'], 1):>9}{fmt(s['td_tail'], 2):>10}")


if __name__ == "__main__":
    main()
