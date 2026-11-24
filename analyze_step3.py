#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_step3.py — フェーズ3 ステップ3（500k 時点のリセット介入）の npz 読み取り

run_step3_intervention_halfcheetah.py が出した *_step3_{condition}.npz
（pure_sac / fixed_prior × control / critic / buffer / actor、計8本。
追加の組み合わせ条件 critic_buffer の2本も同じ書式で読める）を読み、
手法ごとに条件を横並びで比べる。

なぜ analyze_critic_probe.py をそのまま使わないのか:
- ステップ3の npz は 500k〜1M の継続分だけで、timesteps は 505k〜1M。
  ステップ2用スクリプトの区間（0–100k / 100–500k）は空になり、
  位置で割るログ（alpha_logs / pi_entropies など）も 0〜1M に
  引き伸ばされてずれる。ここでは区間を 500k 起点に取り直し、
  位置ログは [開始 step, 終了 step] の区間に線形に割り当てる。

判定（ステップ3設計_500kリセット介入_2026-11-23.md、実行前に固定）:
- 継続の最終区間（900k–1M）平均リターンが 4,000 以上なら「救済」。
- 救済された部品の読み分け: critic → 価値関数の側の固着、
  buffer → データの循環、actor → 方策側の固着、
  critic_buffer → 価値とデータの結びつき。
  どれでも救済されなければ局所最適の読みが強まる。

使い方（keisanB、rl-motion 直下）:
  python3 analyze_step3.py npz_logs_step3_halfcheetah/*_step3_*.npz
  python3 analyze_step3.py npz_logs_step3_halfcheetah/*_step3_*.npz --plot
  --plot を付けると plots_step3/ に手法ごとのリターン曲線（4条件重ね）を出す。
"""
import sys
import os
import numpy as np

# critic_buffer を critic より先に置くこと。parse_name は部分一致で探すため、
# 順序が逆だと *_step3_critic_buffer.npz が critic と誤読される。
CONDITIONS = ["control", "critic_buffer", "critic", "buffer", "actor"]
ALGOS = ["pure_sac", "fixed_prior"]
RESCUE_LINE = 4000.0
# 評価リターンとプローブ記録は絶対 step 基準の区間
WINDOWS = [(500_000, 750_000), (750_000, 900_000), (900_000, 1_000_000)]
WINDOW_LABELS = ["500-750k", "750-900k", "900k-1M"]

READING = {
    "critic": "価値関数の側の固着",
    "buffer": "データの循環",
    "actor": "方策側の固着",
    "critic_buffer": "価値とデータの結びつき",
}


def fmt(x, nd=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "  n/a "
    return f"{x:.{nd}f}"


def wmean_by_steps(steps, values, lo, hi):
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    m = (steps >= lo) & (steps < hi)
    if not m.any() or np.all(np.isnan(values[m])):
        return float("nan")
    return float(np.nanmean(values[m]))


def wmean_positional_interval(values, start, end, lo, hi):
    """継続分だけ切り出されたログ配列を [start, end] に線形に割り当てて区間平均する。"""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0 or end <= start:
        return float("nan")
    i0 = int(n * (lo - start) / (end - start))
    i1 = int(n * (hi - start) / (end - start))
    i0 = max(0, min(i0, n - 1))
    i1 = max(i0 + 1, min(i1, n))
    seg = values[i0:i1]
    if np.all(np.isnan(seg)):
        return float("nan")
    return float(np.nanmean(seg))


def value_near_step(steps, values, target):
    steps = np.asarray(steps, dtype=float)
    values = np.asarray(values, dtype=float)
    if len(steps) == 0:
        return float("nan")
    return float(values[int(np.argmin(np.abs(steps - target)))])


def parse_name(path):
    base = os.path.basename(path)
    algo = "pure_sac" if base.startswith("pure_sac") else (
        "fixed_prior" if base.startswith("fixed_prior") else "?")
    cond = next((c for c in CONDITIONS if f"_step3_{c}" in base), "?")
    return algo, cond


def analyze(path):
    d = np.load(path)
    keys = set(d.files)
    t = d["timesteps"]
    r = d["returns"]
    pt = d["probe_timesteps"] if "probe_timesteps" in keys else np.array([])
    start = int(t.min()) if len(t) else 500_000
    end = int(t.max()) if len(t) else 1_000_000
    algo, cond = parse_name(path)

    ret = [wmean_by_steps(t, r, lo, hi) for lo, hi in WINDOWS]
    final = ret[2]
    rescued = (not np.isnan(final)) and final >= RESCUE_LINE

    def probe(key, nd_win=2):
        if key not in keys or len(pt) == 0:
            return [float("nan")] * 3
        return [wmean_by_steps(pt, d[key], lo, hi) for lo, hi in WINDOWS]

    def poslog(key):
        if key not in keys:
            return [float("nan")] * 3
        return [wmean_positional_interval(d[key], start, end, lo, hi)
                for lo, hi in WINDOWS]

    return {
        "path": path, "algo": algo, "cond": cond,
        "start": start, "end": end,
        "n_eval": len(t), "n_probe": len(pt),
        "r_start": value_near_step(t, r, start),
        "ret": ret, "final": final, "rescued": rescued,
        "max_ret": float(np.max(r)) if len(r) else float("nan"),
        "q1": probe("q1_mean"),
        "td": probe("td_abs_mean"),
        "critic_loss": probe("critic_loss_probe"),
        "entropy": poslog("pi_entropies"),
        "alpha": poslog("alpha_logs"),
        "kl": poslog("kl_values"),
        "beta": poslog("beta_logs"),
    }


def print_table(rows):
    for algo in ALGOS:
        group = [s for s in rows if s["algo"] == algo]
        if not group:
            continue
        group.sort(key=lambda s: CONDITIONS.index(s["cond"])
                   if s["cond"] in CONDITIONS else 99)
        label = "SAC" if algo == "pure_sac" else "先輩版ADR"
        print("=" * 78)
        print(f"{label}（seed 8、500k から 1M へ継続）")
        print(f"  {'条件':<8}{'500k起点':>9}{'500-750k':>10}{'750-900k':>10}"
              f"{'最終900k-1M':>12}{'判定':>8}")
        for s in group:
            verdict = "救済" if s["rescued"] else "—"
            print(f"  {s['cond']:<8}{fmt(s['r_start'], 0):>9}"
                  f"{fmt(s['ret'][0], 0):>10}{fmt(s['ret'][1], 0):>10}"
                  f"{fmt(s['final'], 0):>12}{verdict:>8}")
        print()
        print("  終盤（900k–1M）のプローブ値:")
        print(f"  {'条件':<8}{'Q1':>9}{'|TD|':>8}{'critic loss':>13}"
              f"{'entropy':>9}{'alpha':>9}{'beta':>9}")
        for s in group:
            print(f"  {s['cond']:<8}{fmt(s['q1'][2], 1):>9}{fmt(s['td'][2], 2):>8}"
                  f"{fmt(s['critic_loss'][2], 2):>13}{fmt(s['entropy'][2], 2):>9}"
                  f"{fmt(s['alpha'][2], 4):>9}{fmt(s['beta'][2], 4):>9}")
        print()
        rescued = [s["cond"] for s in group if s["rescued"]]
        if rescued:
            for c in rescued:
                print(f"  → {c} で救済（最終区間平均 ≥ {RESCUE_LINE:.0f}）:"
                      f" {READING.get(c, '')} の読みが強まる。")
        else:
            print(f"  → どの条件でも救済なし（最終区間平均がすべて {RESCUE_LINE:.0f} 未満）:"
                  " 局所最適の読みが強まる。")
        print()


def plot_returns(rows, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for algo in ALGOS:
        group = [s for s in rows if s["algo"] == algo]
        if not group:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        for s in sorted(group, key=lambda x: CONDITIONS.index(x["cond"])
                        if x["cond"] in CONDITIONS else 99):
            d = np.load(s["path"])
            ax.plot(d["timesteps"], d["returns"], lw=0.9,
                    label=f"{s['cond']}" + (" (rescued)" if s["rescued"] else ""))
        ax.axhline(RESCUE_LINE, color="gray", ls="--", lw=0.8,
                   label=f"rescue line {RESCUE_LINE:.0f}")
        ax.set_title(f"Step3 returns: {algo} seed 8 (500k -> 1M)")
        ax.set_xlabel("steps")
        ax.set_ylabel("eval return")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        out = os.path.join(outdir, f"step3_returns_{algo}.png")
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  saved {out}")


def main():
    raw = sys.argv[1:]
    do_plot = "--plot" in raw
    args = [a for a in raw if not a.startswith("--")]
    if not args:
        print(__doc__)
        sys.exit(1)
    # 試し回しの npz（*_smoke.npz）と状態 npz（*_step3state.npz）は対象外。
    # ワイルドカードで拾うと混ざるので、条件として読めたものだけに絞る。
    args = [a for a in args
            if "_smoke" not in os.path.basename(a)
            and parse_name(a)[1] in CONDITIONS]
    if not args:
        print("対象の npz がありません（*_step3_{条件}.npz を指定してください）")
        sys.exit(1)
    rows = [analyze(p) for p in args]
    for s in rows:
        print(f"{s['path']}: eval {s['n_eval']} 点 "
              f"({s['start']:,}–{s['end']:,}), probe {s['n_probe']} 点")
    print()
    print_table(rows)
    if do_plot:
        plot_returns(rows, "plots_step3")


if __name__ == "__main__":
    main()
