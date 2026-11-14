# -*- coding: utf-8 -*-
"""
確認ラン判定に必要な数字の収集と、診断用 npz の間引き抽出を行うスクリプト。

やること:
  1. 遅延アンカー単体・先輩版（FixedPrior）・素の SAC の AUC を表示する
     - AUC(0-500k): 確認ランの判定（条件1）に使う
     - AUC(0-3M): 論文の必須報告項目（3手法の本実験分）
  2. 遅延アンカー＋DroQ 確認ラン（500k × 3シード）の AUC(0-500k) を表示する
  3. 確認ラン npz から診断項目を [::100] に間引いた
     droq_check_seed{s}_extract.npz をリポジトリ直下に作る（条件3の確認用）

npz のキー名はファイルごとに違う可能性があるため、timesteps / returns に
相当する配列はキー名と形から自動判定する。判定に使ったキーは表示する。
NumPy 2.x では np.trapz が無いので np.trapezoid を使う（1.x にも対応）。
"""

import glob

import numpy as np

_trap = getattr(np, "trapezoid", None) or np.trapz


def pick_eval_series(d):
    """npz から (timesteps, returns) らしい配列の組を取り出す。"""
    keys = list(d.files)
    t_keys = [k for k in keys if "time" in k.lower()]
    r_keys = [k for k in keys if ("return" in k.lower() or "reward" in k.lower())]
    # まず名前で決める
    for tk in t_keys:
        for rk in r_keys:
            if len(d[tk]) == len(d[rk]) and len(d[tk]) >= 10:
                return tk, rk, d[tk], d[rk]
    # 名前で決まらなければ、単調増加で最長の配列を timesteps とみなす
    best = None
    for k in keys:
        a = np.asarray(d[k]).ravel()
        if a.dtype.kind not in "fiu" or len(a) < 10:
            continue
        if np.all(np.diff(a.astype(float)) > 0):
            if best is None or len(a) > len(d[best]):
                best = k
    if best is not None:
        t = d[best]
        for k in keys:
            if k != best and len(np.asarray(d[k]).ravel()) == len(t):
                return best, k, t, d[k]
    raise KeyError(f"timesteps/returns らしい配列が見つかりません: keys={keys}")


def auc_upto(t, r, upto):
    m = np.asarray(t) <= upto
    return _trap(np.asarray(r)[m], np.asarray(t)[m])


def report(label, pattern, windows=(500_000, 3_000_000)):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[{label}] ファイルが見つかりません: {pattern}")
        return
    for p in files:
        d = np.load(p)
        tk, rk, t, r = pick_eval_series(d)
        name = p.split("/")[-1]
        parts = [f"AUC(0-{w // 1000}k)={auc_upto(t, r, w):,.0f}" for w in windows]
        print(f"[{label}] {name} (keys: {tk}/{rk}) " + " | ".join(parts))


def main():
    print("=== 本実験 3手法（3M steps） ===")
    report("delayed", "./npz_logs_delayed_anchor_halfcheetah/*.npz")
    report("pure_sac", "./npz_logs_pure_sac_halfcheetah/*.npz")
    report("fixedprior", "./npz_logs_fixed_prior_halfcheetah/*.npz")

    print()
    print("=== 確認ラン（遅延アンカー＋DroQ, 500k steps） ===")
    report("delayed+droq", "./npz_logs_delayed_droq_check/*.npz", windows=(500_000,))

    print()
    print("=== 診断用 extract の作成 ===")
    for s in range(3):
        src = f"./npz_logs_delayed_droq_check/delayed_droq_seed{s}.npz"
        try:
            d = np.load(src)
        except FileNotFoundError:
            print(f"extract seed{s}: 元ファイルなし ({src})")
            continue
        tk, rk, _t, _r = pick_eval_series(d)
        out = {"timesteps": d[tk], "returns": d[rk]}
        for k in d.files:
            if k not in (tk, rk):
                out[k] = d[k][::100]
        dst = f"droq_check_seed{s}_extract.npz"
        np.savez(dst, **out)
        print(f"extract seed{s}: {dst} を作成 (keys: {sorted(out.keys())})")


if __name__ == "__main__":
    main()
