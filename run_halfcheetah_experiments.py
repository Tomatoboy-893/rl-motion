"""
遅延アンカー版 ADR の実験スクリプト（HalfCheetah-v5）
====================================================

背景
----
Humanoid-v5 の本実験は 2 シード完了時点で打ち切りました
（理由と時点データは GOAL.md の打ち切り記録を参照）。
本スクリプトは、環境を先輩の ADR が +40% を出した
HalfCheetah-v5 に変えて、同じ手法を検証するためのものです。

仮説: HalfCheetah はシード間のばらつきが大きく SAC が崩れる
余地があるため、「崩壊を防げば平均が上がる」という遅延アンカーの
勝ち筋が Humanoid より働く可能性がある。

比較する2つ
------------------------------------------------------------------
1. delayed_anchor : 提案手法（事前分布＝少し前のポリシー）
2. pure_sac       : 素の SAC（同一学習ループの PureSACWithLogging）

条件: HalfCheetah-v5 / batch_size=256 / 3,000,000 steps / 5 seeds
  - batch_size=256 は先輩の実行条件（SB3 既定）に合わせたもの
  - target_kl=0.05/次元 は Humanoid 版から据え置き
    （行動次元 6 なので目標は合計 0.30 になる）
評価: 5,000 steps ごとに 5 エピソード（決定論的方策）

保存されるログ (npz)
--------------------
共通 : timesteps, returns, pi_entropies, alpha_logs
提案 : kl_values（KL合計）, kl_raw_values（上限適用前のKL）,
       kl_mean_terms（平均項）, kl_std_terms（σ項）,
       beta_logs（βそのものの値。log ではありません）

★ 最初の seed 0 が、そのまま所要時間の計測を兼ねます。
  1ラン終わるごとに npz が保存されるので、seed 0 完了時点で
  ペースと kl_values の実測を確認し、続行判断ができます。

★ target_kl についての注意
------------------------------------------------------------------
0.05/次元 は Humanoid での実測に合わせた値で、HalfCheetah でも
そのまま妥当とは限りません。seed 0 の kl_values が目標（合計
0.30）に張り付くか、β が下限に落ちないかを確認してください。
"""

import os

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_models import PureSACWithLogging
from sac_delayed_anchor import SACWithDelayedAnchorPrior


class FixedUnifiedReturnCallback(EvalCallback):
    """評価周期（eval_freq）のタイミングでのみ重複なくデータを保存する修正コールバック
    （run_delayed_anchor_experiments.py と同じもの）"""

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


def run_experiment(algo_type, seed, total_timesteps=3_000_000, batch_size=256):
    save_dir = f"./npz_logs_{algo_type}_halfcheetah"
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

    if algo_type == "delayed_anchor":
        model = SACWithDelayedAnchorPrior(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            beta_kl=0.01,      # β の初期値
            beta_lr=1e-3,      # β の自動調整の学習率
            target_kl=0.05,    # KL の目標値（1次元あたり）★実測を見て要調整
            anchor_tau=0.005,  # アンカー（少し前の自分）の追従係数
            seed=seed,
            verbose=0,
        )
    elif algo_type == "pure_sac":
        model = PureSACWithLogging(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError(f"unknown algo_type: {algo_type}")

    print(f"[{algo_type.upper()}] Starting Seed {seed} (batch_size={batch_size})...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    file_path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}.npz"

    # 両アルゴリズム共通で保存する項目
    save_dict = {
        "timesteps": t,
        "returns": r,
        "pi_entropies": np.array(model.pi_entropies),
        "alpha_logs": np.array(model.alpha_logs),
    }

    # 提案手法固有の保存項目（KL は内訳も残す：平均項／σ項の切り分け用）
    if algo_type == "delayed_anchor":
        save_dict.update({
            "kl_values": np.array(model.kl_values),
            "kl_raw_values": np.array(model.kl_raw_values),
            "kl_mean_terms": np.array(model.kl_mean_terms),
            "kl_std_terms": np.array(model.kl_std_terms),
            "beta_logs": np.array(model.beta_logs),
        })

    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")


if __name__ == "__main__":
    # ============================================================
    # 本実験用設定（★現在はこちらが有効）
    # ------------------------------------------------------------
    # 提案手法と Vanilla SAC を 5 シードずつ、3,000,000 steps 回します。
    # seed 0 の完了時点で所要時間の見通しが立つので、
    # そこで一度ログと npz を確認してください。
    # ============================================================
    TOTAL_STEPS = 3_000_000

    # 提案手法と Vanilla SAC の全5シード（batch_size=256）を一括実行
    for algo in ["delayed_anchor", "pure_sac"]:
        for seed in range(5):
            print(f"\n==========================================")
            print(f" Running: {algo} | Seed: {seed} | BS: 256 | HalfCheetah-v5")
            print(f"==========================================")
            run_experiment(algo, seed, total_timesteps=TOTAL_STEPS, batch_size=256)

    # ============================================================
    # 試し回し用設定（確認用に残してあります。通常は使いません）
    # ============================================================
    # TOTAL_STEPS = 100_000
    # print(f"\n==========================================")
    # print(f" Running (TRIAL): delayed_anchor | Seed: 0 | BS: 256 | HalfCheetah-v5")
    # print(f"==========================================")
    # run_experiment("delayed_anchor", 0, total_timesteps=TOTAL_STEPS, batch_size=256)
