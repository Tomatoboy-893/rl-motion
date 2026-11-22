#!/usr/bin/env python3
"""崩壊シードの切り分け診断: beta死か、それ以外か.

背景 (2026-11-20 のチャットでの議論):
- 先輩版 seed 6 は最終区間平均 約2,571 で初の崩壊。候補は2つ:
  H1 beta死: KL が目標に届かず beta が下がり続け、ADR が静かに切れる。
      定量的な盲点: sigma~1, mu~0 の方策は先行分布 N(0, 2^2) に対する
      KL が次元あたり約0.32、合計 約1.9 しかなく目標 6.0 を大きく下回る。
      「這う歩行に固着していても KL の意味では事前分布に近い」状態がありうる。
  H2 beta生存・KL目標通りだが、クリティック側の失敗で進まない。
      ADR はアクター側だけなので、クリティック側の崩壊には届かない。
- このスクリプトは npz の beta / KL / エントロピーの時系列を区間ごとに集計し、
  どちらの仮説と整合するかを判定する。最終方策の分布そのものはログにないので、
  「固着した方策がどれだけ狭いか」はエントロピーの時系列で追う。

使い方:
  python3 diagnose_collapse_seed.py <npz> [--target-kl 6.0] [--label 先輩版seed6]
  # 先輩版 seed 6 の npz は keisanB の ./npz_logs_fixed_prior_halfcheetah にある。
  # 例: scp -P 2022 nuida_23rd@133.14.141.82:/home/nuida_23rd/rl-motion/npz_logs_fixed_prior_halfcheetah/<ファイル名> .
  # (ファイル名は現地で ls して確認すること。推測で決め打ちしない)
  # 遅延アンカー版なら --target-kl 0.30、SAC なら KL/beta のキーが無いので
  # エントロピーとリターンだけ見る (キー欠損時はその旨を表示して続行する)。

判定ルール (区間 2M-3M の平均で見る。しきい値は目安で、数字と併せて読むこと):
  - beta_dead:  終盤 beta 平均 < 0.01 かつ KL 平均 < 目標の 50%  -> H1 と整合
  - beta_alive: 終盤 KL 平均が目標の 80-120% かつ beta 平均 > 0.01 -> H2 と整合
    (beta の 0.01 は「死んでいる」かどうかの境目。先輩版の健全時の beta は
    終盤 約0.38、遅延版の崩壊 seed 1 でも終盤 約0.049 あり、KL が目標通りなら
    beta は死んでいない。0.01 という値は死活の境目として置いた目安。)
  - それ以外:   中間 (beta が弱っている途中、KL が目標付近をうろうろ等)。断定しない。
  注意: 遅延アンカー seed 1 (崩壊) でも KL は目標通り・beta 生存だった。
  したがって「beta生存」は「崩壊していない」ことの証明にはならない。
  逆に「beta死 + KL が目標を大きく下回る」は H1 の強い証拠になる。
- α (alpha_logs) も区間ごとに集計する (2026-11-21 追加。アクター側の最後の
  未確認量)。alpha_logs に入っているのは log α ではなく α そのもの
  (2026-11-07 に sac_models.py で確認)。α の正常範囲に固定のしきい値は
  設けない。α がほぼ 0 に張り付く、桁違いに跳ねるといった極端な動きだけを
  アクター温度の異常の目安とし、ふつうは健全シードとの並べ比べで読む。
"""

import argparse
import sys

import numpy as np

WINDOWS = [
    ("0-100k", 0, 100_000),
    ("100-500k", 100_000, 500_000),
    ("500k-1M", 500_000, 1_000_000),
    ("1-2M", 1_000_000, 2_000_000),
    ("2-3M", 2_000_000, 3_000_000),
]

# kl/beta/entropy は 100 steps ごとに記録される (29999 点 / 3M steps) 前提で、
# 区間のインデックスを steps/100 で換算する。長さが違う場合は線形に割り振る。


def window_slice(n, lo, hi, total_steps=3_000_000):
    i0 = int(round(lo / total_steps * n))
    i1 = int(round(hi / total_steps * n))
    return slice(max(0, i0), min(n, i1))


def mean_or_nan(a):
    return float(np.nanmean(a)) if a is not None and len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", help="診断する npz ファイル")
    ap.add_argument("--target-kl", type=float, default=6.0,
                    help="KL 目標 (合計。先輩版 6.0、遅延アンカー版 0.30)")
    ap.add_argument("--label", default="", help="表示用のラベル")
    args = ap.parse_args()

    try:
        d = np.load(args.npz)
    except Exception as e:
        print(f"npz を開けませんでした: {e}", file=sys.stderr)
        return 1

    print(f"== {args.label or args.npz}  (target KL 合計 {args.target_kl})")
    print(f"keys: {list(d.keys())}")

    get = lambda k: np.asarray(d[k], dtype=float) if k in d.files else None
    kl, kl_raw = get("kl_values"), get("kl_raw_values")
    kl_mean, kl_std = get("kl_mean_terms"), get("kl_std_terms")
    beta, ent = get("beta_logs"), get("pi_entropies")
    alpha = get("alpha_logs")
    returns, steps = get("returns"), get("timesteps")

    if kl is None and kl_raw is None and beta is None:
        print("KL/beta のキーがありません (SAC の npz の可能性)。"
              "エントロピーとリターンだけを表示します。")

    header = (f"{'区間':>10} | {'return':>9} | {'KL':>7} | {'KL_raw':>7} | "
              f"{'KL平均項':>8} | {'KLσ項':>7} | {'beta':>7} | {'entropy':>7} | {'alpha':>7}")
    print(header)
    print("-" * len(header))
    late = {}
    for name, lo, hi in WINDOWS:
        row = {}
        if returns is not None and steps is not None:
            m = (steps >= lo) & (steps < hi)
            row["ret"] = float(np.mean(returns[m])) if m.any() else float("nan")
        else:
            row["ret"] = float("nan")
        for key, arr in [("kl", kl), ("raw", kl_raw), ("m", kl_mean),
                         ("s", kl_std), ("b", beta), ("e", ent), ("a", alpha)]:
            row[key] = mean_or_nan(arr[window_slice(len(arr), lo, hi)]) if arr is not None else float("nan")
        if name == "2-3M":
            late = row
        print(f"{name:>10} | {row['ret']:9.0f} | {row['kl']:7.3f} | {row['raw']:7.3f} | "
              f"{row['m']:8.3f} | {row['s']:7.3f} | {row['b']:7.4f} | {row['e']:7.3f} | {row['a']:7.4f}")

    print()
    print("--- 判定 (2M-3M 区間の平均) ---")
    print(f"KL 平均 {late.get('kl', float('nan')):.3f} / 目標 {args.target_kl} "
          f"(比 {late.get('kl', float('nan')) / args.target_kl * 100:.0f}%)、"
          f"beta 平均 {late.get('b', float('nan')):.4f}、"
          f"エントロピー平均 {late.get('e', float('nan')):.2f}")
    print(f"KL 平均項 {late.get('m', float('nan')):.3f} / KL σ項 {late.get('s', float('nan')):.3f} "
          f"(平均項の割合 {late.get('m', 0) / max(late.get('kl', 1), 1e-9) * 100:.0f}%)")
    a = late.get("a", float("nan"))
    if np.isnan(a):
        print("α 平均 nan (alpha_logs がありません)。")
    else:
        print(f"α 平均 {a:.4f} (アクターの温度。固定のしきい値はなし。"
              "ほぼ 0 に張り付く／桁違いに跳ねる場合のみ異常の目安で、"
              "ふつうは健全シードとの並べ比べで読むこと)")
    b, k = late.get("b", float("nan")), late.get("kl", float("nan"))
    if np.isnan(b) or np.isnan(k):
        print("判定: beta/KL の記録がないため判定できません。")
    elif b < 0.01 and k < 0.5 * args.target_kl:
        print("判定: H1 (beta死) と整合します。KL が目標に届かず beta が死に、"
              "ADR が静かに切れていた可能性が高いです。")
        print("  -> 次に見るべき点: KL がいつ目標を下回り始めたか (上の区間表で確認)。"
              "這う歩行のまま KL が小さい状態 (sigma~1, mu~0 で KL 合計~1.9) に"
              "入った時期と重なるか。")
    elif 0.8 * args.target_kl <= k <= 1.2 * args.target_kl and b > 0.01:
        print("判定: H2 (beta生存) と整合します。KL は目標通りで beta も生きているので、"
              "崩壊の主因は beta死ではなく、クリティック側など別経路の可能性が高いです。")
        print("  -> 次に見るべき点: エントロピーの推移 (方策が狭まったか) と、"
              "Q / critic loss の時系列 (npz に無い場合は学習ログで確認)。")
    else:
        print("判定: 中間です。H1/H2 のどちらとも断定できません。")
        print("  -> beta が弱りつつある途中か、KL が目標付近をうろうろしている状態です。"
              "区間表の推移を併せて読んでください。")
    print()
    print("留保: これは npz の時系列からの切り分けで、因果の確定ではありません。"
          "最終方策の分布そのものはログにないため、固着の狭まりはエントロピーでしか追えません。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
