#!/usr/bin/env python3
"""
項目4（α を戻す介入）の集計
============================
./npz_logs_alpha_halfcheetah/ の npz と、ステップ3の control
（./npz_logs_step3_halfcheetah/）を読み、最終区間（900k–1M）の
評価リターン平均と、α の軌道の要約を出す。

判定（設計メモで事前に固定）: 最終区間平均 4,000 以上で救済。
alpha_fixed では α が全期間目標値のままかも検証する。

使い方:
    python3 analyze_alpha_intervention.py
"""

import os

import numpy as np

ALPHA_DIR = "./npz_logs_alpha_halfcheetah"
STEP3_DIR = "./npz_logs_step3_halfcheetah"
ALGOS = ["pure_sac", "fixed_prior"]
SEED = 8
CONDITIONS = ["alpha_reset", "alpha_fixed"]
ALPHA_TARGETS = {"pure_sac": 0.1581, "fixed_prior": 0.1226}
FINAL_LO, FINAL_HI = 900_000, 1_000_000
RESCUE_LINE = 4000.0


def final_window_mean(timesteps, returns):
    mask = (timesteps >= FINAL_LO) & (timesteps <= FINAL_HI)
    if mask.sum() == 0:
        return None, 0
    return float(returns[mask].mean()), int(mask.sum())


def alpha_summary(alpha_logs):
    """継続区間の α の要約（序盤100点・終盤100点・全体の範囲）。"""
    a = np.asarray(alpha_logs, dtype=float)
    if a.size == 0:
        return None
    head = float(a[:100].mean()) if a.size >= 100 else float(a.mean())
    tail = float(a[-100:].mean()) if a.size >= 100 else float(a.mean())
    return {
        "n": int(a.size),
        "head100": head,
        "tail100": tail,
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
    }


def load_npz(path):
    if not os.path.exists(path):
        return None
    return np.load(path)


def main():
    print("=== 項目4: α を戻す介入の集計（最終区間 900k–1M、救済ライン 4,000） ===")
    for algo in ALGOS:
        print(f"\n--- {algo} (seed {SEED}) ---")
        # 対照: ステップ3の control
        ctrl_path = os.path.join(
            STEP3_DIR, f"{algo}_halfcheetah_seed{SEED}_step3_control.npz")
        ctrl = load_npz(ctrl_path)
        if ctrl is None:
            print(f" control (step3): ファイルなし: {ctrl_path}")
        else:
            mean, n = final_window_mean(ctrl["timesteps"], ctrl["returns"])
            print(f" control (step3): 最終区間平均 = {mean:,.1f} (n={n})")

        for condition in CONDITIONS:
            path = os.path.join(
                ALPHA_DIR, f"{algo}_halfcheetah_seed{SEED}_{condition}.npz")
            data = load_npz(path)
            if data is None:
                print(f" {condition}: ファイルなし: {path}")
                continue
            mean, n = final_window_mean(data["timesteps"], data["returns"])
            verdict = "救済" if (mean is not None and mean >= RESCUE_LINE) else "救済されず"
            print(f" {condition}: 最終区間平均 = {mean:,.1f} (n={n}) -> {verdict}")
            s = alpha_summary(data["alpha_logs"])
            if s is not None:
                print(f"   alpha: n={s['n']} 序盤100点平均={s['head100']:.4f} "
                      f"終盤100点平均={s['tail100']:.4f} "
                      f"min={s['min']:.4f} max={s['max']:.4f} 平均={s['mean']:.4f}")
                if condition == "alpha_fixed":
                    target = ALPHA_TARGETS[algo]
                    dev = max(abs(s["min"] - target), abs(s["max"] - target))
                    ok = dev < 1e-3
                    print(f"   alpha_fixed 検証: 目標 {target} からの最大のずれ = "
                          f"{dev:.6f} -> {'固定を確認' if ok else '警告: 固定されていない'}")
            if "beta_logs" in data:
                b = np.asarray(data["beta_logs"], dtype=float)
                if b.size:
                    print(f"   beta（保持）: 序盤100点平均={b[:100].mean():.4f} "
                          f"終盤100点平均={b[-100:].mean():.4f}")


if __name__ == "__main__":
    main()
