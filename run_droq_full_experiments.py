"""
遅延アンカー + DroQ 流クリティック：本ラン用スクリプト（HalfCheetah-v5）
====================================================================

位置づけ
--------
確認ラン（500k steps × 3シード、run_droq_check_experiments.py）の
結果を受けて行う本ランです。モデルは確認ランとまったく同じ
sac_delayed_anchor_droq.py の SACWithDelayedAnchorDroQ
（変更はクリティック側だけ）で、条件も確認ランと同一です。
違うのは規模（3M steps × 5シード）と保存先だけです。

確認ランの判定（2026-11-13 に固定した基準、2026-11-14 に判定）
----------------------------------------------------------------
- 条件(1) AUC（0〜500k）3シード平均：併用 約2.137e9 が
  遅延アンカー単体の約1.889e9 を +13.2% 上回った → 満たす
- 条件(2) 500k 時点で這うシード数：併用 1本（seed 1）＝単体 1本
  → 増えていない → 満たす
- 条件(3) 診断：3シードとも KL 終盤平均 0.297〜0.299（目標 0.30）、
  β は下限に張り付かず生存、NaN・Q／critic loss の発散なし → 満たす
→ 判定：本ランへ進める

本ランで答える問い
------------------
最終区間（2.5M〜3.0M）の IQM で、先輩版 FixedPrior の
約12,726 を上回れるか。比較条件は本実験（遅延アンカー単体・
素の SAC・先輩版）と同一にそろえてあり、最終区間平均・IQM・
崩壊（最終区間平均 4,000 未満）の物差しもそのまま使えます。

条件（確認ラン・本実験と同一）
------------------------------
- 環境: HalfCheetah-v5 / batch_size=256
- 遅延アンカー側: beta_kl=0.01、beta_lr=1e-3、target_kl=0.05/次元
  （合計 0.30）、anchor_tau=0.005
- DroQ 側: ドロップアウト率 0.01、UTD=5（gradient_steps=5）、
  方策 Q は min（policy_q_agg="min"）
- 評価: 5,000 steps ごとに 5 エピソード（決定論的方策）

保存先（確認ランと分けること。上書き防止）
------------------------------------------
./npz_logs_delayed_droq_full/delayed_droq_full_seed{seed}.npz
中身は確認ランと同一形式（timesteps, returns, 診断ログ一式）。

所要時間の見通し（確認ランの実測から）
--------------------------------------
確認ラン 500k steps は 7.2〜8.6 時間／シード（16.2〜19.3 steps/秒）
でした。3M steps では約 43〜52 時間／シード、5シード逐次で
約 9〜11 日の見込みです。
"""

import argparse
import os
import time

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_delayed_anchor_droq import DroQSACPolicy, SACWithDelayedAnchorDroQ


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


def run_experiment(seed, total_timesteps, batch_size=256,
                   save_dir="./npz_logs_delayed_droq_full",
                   gradient_steps=5, policy_q_agg="min"):
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

    model = SACWithDelayedAnchorDroQ(
        DroQSACPolicy,
        train_env,
        batch_size=batch_size,
        beta_kl=0.01,        # β の初期値（本実験・確認ランと同一）
        beta_lr=1e-3,        # β の自動調整の学習率（同一）
        target_kl=0.05,      # KL の目標値（1次元あたり。同一）
        anchor_tau=0.005,    # アンカーの追従係数（同一）
        gradient_steps=gradient_steps,  # UTD（DroQ 側の変更）
        policy_q_agg=policy_q_agg,      # 方策 Q のまとめ方（既定 min）
        seed=seed,
        verbose=0,
    )

    print(f"[DELAYED+DroQ FULL] Starting Seed {seed} "
          f"(batch_size={batch_size}, steps={total_timesteps}, "
          f"UTD={gradient_steps}, policy_q_agg={policy_q_agg})...")
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, callback=callback,
                progress_bar=True)
    elapsed = time.time() - t0

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    file_path = f"{save_dir}/delayed_droq_full_seed{seed}.npz"
    save_dict = {
        "timesteps": t,
        "returns": r,
        "pi_entropies": np.array(model.pi_entropies),
        "alpha_logs": np.array(model.alpha_logs),
        "kl_values": np.array(model.kl_values),
        "kl_raw_values": np.array(model.kl_raw_values),
        "kl_mean_terms": np.array(model.kl_mean_terms),
        "kl_std_terms": np.array(model.kl_std_terms),
        "beta_logs": np.array(model.beta_logs),
        "brake_factors": np.array(model.brake_factors),
        "critic_losses": np.array(model.critic_losses),
        "q_means": np.array(model.q_means),
    }
    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")
    print(f" 所要時間 {elapsed / 3600:.2f} 時間 "
          f"({total_timesteps / max(elapsed, 1e-9):.1f} steps/秒)")
    return model, elapsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="遅延アンカー + DroQ 本ラン（3M steps × 5シード）")
    ap.add_argument("--steps", type=int, default=3_000_000,
                    help="ステップ数（既定 3,000,000）")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="シード（既定 0,1,2,3,4）")
    ap.add_argument("--policy-q-agg", type=str, default="min",
                    choices=["min", "mean"],
                    help="方策 Q のまとめ方（既定 min）")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    for seed in seeds:
        print(f"\n==========================================")
        print(f" Running (FULL): delayed+droq | Seed: {seed} | "
              f"BS: 256 | HalfCheetah-v5")
        print(f"==========================================")
        run_experiment(seed, total_timesteps=args.steps,
                       batch_size=256,
                       policy_q_agg=args.policy_q_agg)
