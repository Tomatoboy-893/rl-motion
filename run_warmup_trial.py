"""
遅延アンカー版 ADR「ウォームアップ版」の実験スクリプト（Humanoid-v5）
====================================================================

ウォームアップ版とは
--------------------
遅延アンカー版（sac_delayed_anchor.py）のブレーキは、序盤から
常時作動するため、立ち上がりが SAC の約半分の速さになることが
同シード比較で分かりました（本実験 seed 0 の途中経過より）。
一方、リターンの崩れが実際に起きるのは中盤以降です。

そこでこの版では、ブレーキの実効係数を
  - 100,000 steps までは 0（ブレーキなし＝ほぼ素の SAC の動き）
  - 100,000 → 600,000 steps で 0 から 1 へ線形に増加
  - 600,000 steps 以降は 1（従来の遅延アンカー版と同じ強さ）
と段階的に投入します。「安全装置は普段は邪魔をせず、
危ない時期に作動する」という本来の狙いに合わせた版です。
※ 区間の数値は、前回実験で崩れが 1.5M〜2.7M に集中していた
   ことと、SAC の立ち上がりが概ね 500k までに起きることに
   基づく目安です（今後の結果次第で調整対象）。

比較のしかた
------------
このスクリプトの 500k 試し回し（seed 0）と、
  - 本実験の遅延アンカー版 seed 0（npz_logs_delayed_anchor_bs64）
  - 前回実験の SAC seed 0
の 0〜500k 区間を比べ、「立ち上がりが SAC 並みに戻ったか」を
確認するのが目的です。

保存されるログ (npz)
--------------------
timesteps, returns, pi_entropies, alpha_logs,
kl_values, kl_raw_values, kl_mean_terms, kl_std_terms,
beta_logs, brake_factors（ブレーキ係数の実績値）
"""

import os

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_delayed_anchor import SACWithDelayedAnchorPrior


class FixedUnifiedReturnCallback(EvalCallback):
    """評価周期（eval_freq）のタイミングでのみ重複なくデータを保存する修正コールバック
    （run_bs64_experiments.py と同じもの）"""

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


# ウォームアップの区間（環境ステップ数）
WARMUP_START = 100_000   # ここまでブレーキ係数 0
WARMUP_END = 600_000     # ここで係数 1（フル作動）


def run_experiment(seed, total_timesteps=500_000, batch_size=64):
    algo_type = "delayed_warmup"
    save_dir = f"./npz_logs_{algo_type}_bs64"
    os.makedirs(save_dir, exist_ok=True)

    train_env = make_vec_env("Humanoid-v5", n_envs=1, seed=seed)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=seed + 1000)

    callback = FixedUnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = SACWithDelayedAnchorPrior(
        "MlpPolicy",
        train_env,
        batch_size=batch_size,
        beta_kl=0.01,      # β の初期値
        beta_lr=1e-3,      # β の自動調整の学習率
        target_kl=0.05,    # KL の目標値（1次元あたり）
        anchor_tau=0.005,  # アンカー（少し前の自分）の追従係数
        brake_warmup_start=WARMUP_START,  # ブレーキ投入の開始
        brake_warmup_end=WARMUP_END,      # ブレーキがフルになる時点
        seed=seed,
        verbose=0,
    )

    print(f"[DELAYED_WARMUP] Starting Seed {seed} (batch_size={batch_size})...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    train_env.close()
    eval_env.close()

    file_path = f"{save_dir}/{algo_type}_bs64_seed{seed}.npz"
    np.savez(
        file_path,
        timesteps=np.array(callback.timesteps),
        returns=np.array(callback.episode_returns),
        pi_entropies=np.array(model.pi_entropies),
        alpha_logs=np.array(model.alpha_logs),
        kl_values=np.array(model.kl_values),
        kl_raw_values=np.array(model.kl_raw_values),
        kl_mean_terms=np.array(model.kl_mean_terms),
        kl_std_terms=np.array(model.kl_std_terms),
        beta_logs=np.array(model.beta_logs),
        brake_factors=np.array(model.brake_factors),
    )
    print(f" Saved to {file_path}")


if __name__ == "__main__":
    # ============================================================
    # 試し回し用設定（★現在はこちらが有効）
    # ------------------------------------------------------------
    # 目的：0〜500k 区間の立ち上がりが、素の SAC seed 0 並みに
    # 戻るかを確認すること。seed 0 のみ・500,000 steps。
    # ============================================================
    print(f"\n==========================================")
    print(f" Running (TRIAL): delayed_warmup | Seed: 0 | BS: 64")
    print(f"==========================================")
    run_experiment(0, total_timesteps=500_000, batch_size=64)

    # ============================================================
    # 本実験用設定（試し回しで立ち上がりの回復が確認できたら、
    # 上のブロックと入れ替えて使ってください）
    # ============================================================
    # for seed in range(5):
    #     print(f"\n==========================================")
    #     print(f" Running: delayed_warmup | Seed: {seed} | BS: 64")
    #     print(f"==========================================")
    #     run_experiment(seed, total_timesteps=3_000_000, batch_size=64)
