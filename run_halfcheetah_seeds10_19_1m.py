"""
SAC／先輩版 ADR のシード増設ラン（HalfCheetah-v5、seed 10〜19、1M steps）
====================================================================

位置づけ（フェーズ1.5：傾向確認のための追加分）
------------------------------------------------
2026-11-23 の検討でユーザーが提案し、方向性を確認した追加ラン。
10 シードでは「seed への依存が強すぎる」という感覚があり、
新しい seed で傾向を確かめるのが目的。

問うのは次の2点だけ。どちらも、見る前に固定する。
  (1) 這い型の崩壊は新しい seed でも起きるか（発生の傾向）
  (2) α の序盤の分離（100〜500k で崩壊／健全が分かれる）は、
      まだ見ていない seed でも当たるか（後付けでない検証）

問わないこと（1M では測れないことが 20 ランの実績で分かっているため）
  - 最終リターンや IQM（3M の集計値）
  - 頭打ち型（先輩版 seed 1）や遅れて加速する型（先輩版 seed 2）の分類
  - 途中の一時的崩壊（先輩版 seed 0 は約 2.1M で起きた）
  このため、追加ランは 3M の 10 シード集計には混ぜない。
  別枠の「傾向確認」として扱う。

判定の線引き（走らせる前に固定する。結果を見てから動かさない）
------------------------------------------------------------------
- 崩壊（這い型）：500k〜1M 区間の評価リターン平均が 4,000 未満。
  卒論の崩壊ライン（最終区間平均 4,000 未満）と同じ値を使う。
  20 ランの実績では、1M 時点で這い型は約 2,500 以下、それ以外は
  約 7,000 以上と二極化していた。4,000〜7,000 の中間が出た場合は
  「判定保留」とし、無理に二分しない。
- α の検証：100k〜500k 区間の α 平均を、崩壊群と健全群で比べる。
  しきい値は事後に作らない。「重なりなしに分かれるか」と
  「向き」（健全は上昇・崩壊は下降）を見るだけの記述的確認とする。
  β（先輩版）についても同じ区間で水準を並べる。

条件（既存ランと完全に同一。手法の中身には一切手を入れない）
------------------------------------------------------------------
- 環境: HalfCheetah-v5 / batch_size=256 / 1,000,000 steps
- pure_sac   : PureSACWithLogging（sac_models.py、素の SAC）
- fixed_prior: SACWithFixedPrior（sac_adr_main.py、先輩本人の実装）
  prior_std=2.0 / target_kl=1.0（次元あたり）/ beta_kl=0.01 / beta_lr=1e-3
- 評価: 5,000 steps ごとに 5 エピソード（決定論的方策）
- 初期分布の記録：各ランの学習直前に初期方策の μ・σ を実測して
  ./npz_logs_initdist_halfcheetah/ に保存する（拡張ランと同じ関数）
- 保存先・ファイル名・npz のキーは既存スクリプトと同一
  （./npz_logs_pure_sac_halfcheetah / ./npz_logs_fixed_prior_halfcheetah）。
  seed 10〜19 は既存の seed 0〜9 と衝突しない。

実行順序
------------------------------------------------------------------
seed を外側のループにし、1 シードごとに SAC → 先輩版 の順で回す。
途中で止めても両手法のシード数が揃った状態で残るための配慮
（拡張ランと同じ）。

所要時間の見通し（概算）
------------------------------------------------------------------
1M steps は 1 本あたり約 3 時間。20 本（2手法×10シード）で
約 60 時間＝約 2.5 日の見込み（keisanB で連続運転の場合）。
なお 2M にする場合は TOTAL_STEPS を 2_000_000 に変えるだけ
（1 本約 6 時間、20 本で約 5 日）。その場合は先輩版 seed 2 型の
遅加速も 2M 時点で見えるようになる。

起動前の確認（走らせる前にチェックすること）
------------------------------------------------------------------
- ステップ2のプローブ再走（4本）とステップ3の介入実験
  （単独4条件＋組み合わせ、計10本）は完了済み（2026-11-25）。
  計算機を共有するため、ほかの学習プロセスが走っている最中の
  起動はしない。
- DroQ 本ランは停止する方針（2026-11-18 決定）。
- 起動順はステップ3の後と決まり、2026-11-25 にユーザーが
  実施を決定した。
"""

import os
import sys

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior

# 試し回し（True にすると 10,000 steps・保存先は smoke 用フォルダ）
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"

# 追加シードと尺度。2M にする場合は TOTAL_STEPS だけを変える。
SEEDS = range(10, 20)
TOTAL_STEPS = 1_000_000


class FixedUnifiedReturnCallback(EvalCallback):
    """評価周期（eval_freq）のタイミングでのみ重複なくデータを保存する修正コールバック
    （run_halfcheetah_experiments.py と同じもの）"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.timesteps = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.episode_returns.append(self.last_mean_reward)
                self.timesteps.append(self.num_timesteps)
        return result


def snapshot_initial_distribution(model, algo_type, seed, n_steps=2000):
    """学習前の方策が吐く分布（tanh 前ガウス分布の μ・σ）を実測して保存する。

    学習には一切影響させない。失敗しても例外を外に出さず警告のみ。
    （run_halfcheetah_10seed_extension.py と同じ関数）
    """
    try:
        import torch

        probe_env = gym.make("HalfCheetah-v5")
        obs, _ = probe_env.reset(seed=seed * 1000 + 777)
        states = []
        for _ in range(n_steps):
            states.append(obs)
            obs, _, terminated, truncated, _ = probe_env.step(
                probe_env.action_space.sample()
            )
            if terminated or truncated:
                obs, _ = probe_env.reset()
        probe_env.close()
        states = np.array(states, dtype=np.float32)

        actor = model.policy.actor
        mus, sigmas = [], []
        with torch.no_grad():
            for i in range(0, len(states), 512):
                batch = states[i:i + 512]
                obs_tensor, _ = model.policy.obs_to_tensor(batch)
                mean_actions, log_std, _ = actor.get_action_dist_params(obs_tensor)
                mus.append(mean_actions.cpu().numpy())
                sigmas.append(np.exp(log_std.cpu().numpy()))
        mu = np.concatenate(mus)
        sigma = np.concatenate(sigmas)

        save_dir = "./npz_logs_initdist_halfcheetah"
        if SMOKE:
            save_dir = "./npz_logs_smoke_seeds10_19_1m"
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}_initdist.npz"
        np.savez(
            path,
            mu_samples=mu,
            sigma_samples=sigma,
            mu_mean_per_dim=mu.mean(axis=0),
            mu_std_per_dim=mu.std(axis=0),
            sigma_mean_per_dim=sigma.mean(axis=0),
            sigma_std_per_dim=sigma.std(axis=0),
            sigma_min=float(sigma.min()),
            sigma_max=float(sigma.max()),
        )
        print(f" [initdist] saved {path} "
              f"(sigma mean per dim: {np.round(sigma.mean(axis=0), 4)})")
    except Exception as e:  # 記録の失敗で本ランを止めない
        print(f" [initdist] WARNING: 初期分布の記録に失敗しました（続行します）: {e}")


def run_experiment(algo_type, seed, total_timesteps=1_000_000, batch_size=256):
    save_dir = f"./npz_logs_{algo_type}_halfcheetah"
    if SMOKE:
        save_dir = "./npz_logs_smoke_seeds10_19_1m"
    os.makedirs(save_dir, exist_ok=True)

    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    callback = FixedUnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    if algo_type == "pure_sac":
        model = PureSACWithLogging(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            seed=seed,
            verbose=0,
        )
    elif algo_type == "fixed_prior":
        model = SACWithFixedPrior(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            beta_kl=0.01,    # β の初期値（先輩の設定）
            beta_lr=1e-3,    # β の自動調整の学習率（先輩の設定）
            target_kl=1.0,   # KL の目標値（1次元あたり。内部で×行動次元）
            prior_std=2.0,   # 事前分布の σ（先輩の設定）
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError(f"unknown algo_type: {algo_type}")

    # 学習開始直前に初期分布を記録（学習には影響しない）
    snapshot_initial_distribution(model, algo_type, seed)

    print(f"[{algo_type.upper()}] Starting Seed {seed} (batch_size={batch_size})...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    file_path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}.npz"

    save_dict = {
        "timesteps": t,
        "returns": r,
        "pi_entropies": np.array(model.pi_entropies),
        "alpha_logs": np.array(model.alpha_logs),
    }
    if algo_type == "fixed_prior":
        save_dict.update({
            "kl_values": np.array(model.kl_values),
            "beta_logs": np.array(model.beta_logs),
        })

    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")


def save_1m_summaries():
    """seed 10〜19（1M）が揃った手法について、1M 用の集計 npz を保存する。

    注意：3M の 10 シード集計（save_10seed_summaries）とは別物。
    この追加ランは 3M 集計には混ぜない。
    """
    for algo_type in ("pure_sac", "fixed_prior"):
        save_dir = f"./npz_logs_{algo_type}_halfcheetah"
        all_returns, timesteps = [], None
        ok = True
        for seed in SEEDS:
            path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}.npz"
            if not os.path.exists(path):
                print(f" [集計] {path} が無いため {algo_type} の集計は作りません")
                ok = False
                break
            d = np.load(path)
            if timesteps is None:
                timesteps = np.asarray(d["timesteps"])
            all_returns.append(np.asarray(d["returns"]))
        if ok:
            arr = np.array(all_returns)
            out = f"{save_dir}/{algo_type}_seeds10_19_1m_mean_std.npz"
            np.savez(out, timesteps=timesteps, all_returns=arr,
                     mean=arr.mean(axis=0), std=arr.std(axis=0))
            print(f" Saved summary to {out}")


if __name__ == "__main__":
    if SMOKE:
        # 試し回し: seed 10 の SAC を 10,000 steps だけ（保存先は smoke 用）
        print("=== SMOKE MODE (10,000 steps, seed 10, pure_sac only) ===")
        run_experiment("pure_sac", 10, total_timesteps=10_000, batch_size=256)
    else:
        for seed in SEEDS:
            for algo_type in ("pure_sac", "fixed_prior"):
                print(f"\n==========================================")
                print(f" Running: {algo_type} | Seed: {seed} | BS: 256 | HalfCheetah-v5 | {TOTAL_STEPS} steps")
                print(f"==========================================")
                run_experiment(algo_type, seed, total_timesteps=TOTAL_STEPS, batch_size=256)
        save_1m_summaries()
