"""
遅延アンカー + DroQ 流クリティック：確認ラン用スクリプト（HalfCheetah-v5）
====================================================================

位置づけ
--------
提案手法（遅延アンカー版 ADR）に DroQ 流のクリティック強化
（ドロップアウト＋LayerNorm、UTD=5）を併用し、本ラン（3M steps ×
5シード）に進む価値があるかを、小さなランで先に確かめるための
スクリプトです。モデルは sac_delayed_anchor_droq.py の
SACWithDelayedAnchorDroQ（変更はクリティック側だけ）。

進め方（段階制）
----------------
1. スモークテスト: --smoke（seed 0、10,000 steps）
   見ること:
   - モデルが構築できて学習が回ること（SB3 のバージョン依存の確認）
   - model._n_updates が環境ステップ数の約 5 倍になっていること
     （gradient_steps=5 が本当に効いているか）
   - KL が目標（合計 0.30）付近にいること、β が 0 に落ちていないこと
   - critic loss / Q 値が NaN・発散していないこと
   - 速度（steps/秒）から、本ランの所要時間を見積もれること
2. 確認ラン: 既定（seed 0〜2、500,000 steps）
   比較相手は、すでに完走している遅延アンカー単体の 3M ラン
   （./npz_logs_delayed_anchor_halfcheetah）の同じシードの
   0〜500k 区間です（AUC は npz から後で計算します）。
   判断の基準は、結果を見る前に別途固定します（このスクリプトの
   実行をもって基準を固定したことにはなりません）。

条件
----
- 環境: HalfCheetah-v5 / batch_size=256（本実験と同一）
- 遅延アンカー側: beta_kl=0.01、beta_lr=1e-3、target_kl=0.05/次元
  （合計 0.30）、anchor_tau=0.005（本実験と同一）
- DroQ 側: ドロップアウト率 0.01（DroQSACPolicy のクラス属性）、
  UTD=5（gradient_steps=5）、方策 Q は min（policy_q_agg="min"）
- 評価: 5,000 steps ごとに 5 エピソード（決定論的方策）。
  本実験と同じ形式で保存します。

保存されるログ (npz)
--------------------
確認ラン: ./npz_logs_delayed_droq_check/delayed_droq_seed{seed}.npz
スモーク: ./npz_logs_delayed_droq_smoke/delayed_droq_seed0.npz
共通 : timesteps, returns, pi_entropies, alpha_logs,
       kl_values, kl_raw_values, kl_mean_terms, kl_std_terms,
       beta_logs, brake_factors
追加 : critic_losses, q_means（Q の発散・NaN の監視用）
※ beta_logs は環境ステップごと、その他の学習ログは勾配ステップ
   ごとの記録です（sac_delayed_anchor_droq.py の注意書き参照）。

所要時間の見通し（未測定の推定）
--------------------------------
遅延アンカー単体は 3M steps で約 8時間45分／シードでした。
勾配計算が支配的なら、500k steps・UTD=5 は約 7〜8 時間／シード、
3シードで約 1 日の見込みです。まずスモークで実測してください。
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
                   save_dir="./npz_logs_delayed_droq_check",
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
        beta_kl=0.01,        # β の初期値（本実験と同一）
        beta_lr=1e-3,        # β の自動調整の学習率（本実験と同一）
        target_kl=0.05,      # KL の目標値（1次元あたり。本実験と同一）
        anchor_tau=0.005,    # アンカーの追従係数（本実験と同一）
        gradient_steps=gradient_steps,  # UTD（DroQ 側の変更）
        policy_q_agg=policy_q_agg,      # 方策 Q のまとめ方（既定 min）
        seed=seed,
        verbose=0,
    )

    print(f"[DELAYED+DroQ] Starting Seed {seed} "
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

    file_path = f"{save_dir}/delayed_droq_seed{seed}.npz"
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


def tail_summary(model, label):
    """スモークテスト用の簡易診断（最後の区間の平均を出す）。"""
    def tail(a, frac=0.1):
        if len(a) == 0:
            return float("nan")
        k = max(1, int(len(a) * frac))
        return float(np.mean(a[-k:]))

    print(f" [{label}] _n_updates={model._n_updates} "
          f"(環境ステップ {model.num_timesteps} の "
          f"{model._n_updates / max(model.num_timesteps, 1):.2f} 倍)")
    print(f" [{label}] KL 終盤平均 {tail(model.kl_values):.3f} "
          f"(目標 0.30) / β 終盤平均 {tail(model.beta_logs):.4f}")
    cl = np.array(model.critic_losses, dtype=float)
    qm = np.array(model.q_means, dtype=float)
    print(f" [{label}] critic loss 終盤平均 "
          f"{tail(model.critic_losses):.3f} / Q 終盤平均 "
          f"{tail(model.q_means):.1f} / NaN: loss "
          f"{bool(np.isnan(cl).any())}, Q {bool(np.isnan(qm).any())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="遅延アンカー + DroQ 確認ラン")
    ap.add_argument("--smoke", action="store_true",
                    help="スモークテスト（seed 0、10k steps）")
    ap.add_argument("--steps", type=int, default=500_000,
                    help="確認ランのステップ数（既定 500,000）")
    ap.add_argument("--seeds", type=str, default="0,1,2",
                    help="確認ランのシード（既定 0,1,2）")
    ap.add_argument("--policy-q-agg", type=str, default="min",
                    choices=["min", "mean"],
                    help="方策 Q のまとめ方（既定 min）")
    args = ap.parse_args()

    if args.smoke:
        print("\n==========================================")
        print(" Running (SMOKE): delayed+droq | Seed: 0 | 10k steps")
        print("==========================================")
        model, _ = run_experiment(
            0, total_timesteps=10_000, batch_size=256,
            save_dir="./npz_logs_delayed_droq_smoke",
            policy_q_agg=args.policy_q_agg)
        tail_summary(model, "SMOKE")
    else:
        seeds = [int(s) for s in args.seeds.split(",")]
        for seed in seeds:
            print(f"\n==========================================")
            print(f" Running: delayed+droq | Seed: {seed} | "
                  f"BS: 256 | HalfCheetah-v5")
            print(f"==========================================")
            run_experiment(seed, total_timesteps=args.steps,
                           batch_size=256,
                           policy_q_agg=args.policy_q_agg)