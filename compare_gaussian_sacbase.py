# compare_gaussian_sacbase.py

import os
import time
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

SAVE_DIR = "./npz_logs_halfcheetah"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 1. ロス記録機能を持つ共通のベースクラス
# ==========================================
class SACHistoryLogger(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor_losses_history = []
        self.critic_losses_history = []

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        
        logger_vals = self.logger.name_to_value
        if "train/actor_loss" in logger_vals:
            val = logger_vals["train/actor_loss"]
            if val is not None and not np.isnan(val):
                self.actor_losses_history.append(val)
        if "train/critic_loss" in logger_vals:
            val = logger_vals["train/critic_loss"]
            if val is not None and not np.isnan(val):
                self.critic_losses_history.append(val)

# ==========================================
# 2. 固定事前分布（Gaussian Prior / rho）付き SAC クラス
# ==========================================
class SACWithFixedPrior(SACHistoryLogger):
    def __init__(self, *args, prior_std=1.0, beta_kl=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_std = prior_std
        self.beta_kl = beta_kl


# ==========================================
# 3. 評価・リターン・エントロピー記録用コールバック
# ==========================================
class MetricsCallback(EvalCallback):
    def __init__(self, model_ref, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref
        self.episode_returns = []
        self.timesteps = []
        self.entropies = []
        self.actor_losses = []
        self.critic_losses = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.episode_returns.append(self.last_mean_reward)
                self.timesteps.append(self.num_timesteps)

                # --- 1. エントロピーの計算 ---
                with torch.no_grad():
                    try:
                        replay_data = self.model_ref.replay_buffer.sample(256)
                        obs_tensor = replay_data.observations
                        mean_actions, log_std, kwargs_dist = self.model_ref.actor.get_action_dist_params(obs_tensor)
                        _, log_prob = self.model_ref.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs_dist)
                        entropy = (-log_prob).mean().item()
                    except Exception:
                        entropy = 0.0
                self.entropies.append(entropy)

                # --- 2. ロスの取得（区間平均） ---
                if self.model_ref.actor_losses_history:
                    self.actor_losses.append(float(np.mean(self.model_ref.actor_losses_history)))
                    self.model_ref.actor_losses_history = []
                else:
                    self.actor_losses.append(self.actor_losses[-1] if self.actor_losses else 0.0)

                if self.model_ref.critic_losses_history:
                    self.critic_losses.append(float(np.mean(self.model_ref.critic_losses_history)))
                    self.model_ref.critic_losses_history = []
                else:
                    self.critic_losses.append(self.critic_losses[-1] if self.critic_losses else 0.0)

        return result


# ==========================================
# 4. 実験実行メイン関数
# ==========================================
def run_experiment(algo_type, rho_value=None, seed=0, total_steps=3_000_000):
    print(f"\n=========================================")
    print(f" Starting HalfCheetah: {algo_type.upper()} (rho={rho_value}), Seed={seed}")
    print(f"=========================================")
    
    train_env = make_vec_env("HalfCheetah-v4", n_envs=8, seed=seed)
    eval_env = gym.make("HalfCheetah-v4")
    eval_env.reset(seed=seed)
    
    if algo_type == "gaussian":
        model = SACWithFixedPrior(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            batch_size=256,
            prior_std=rho_value,
            verbose=0,
            device="cuda",
            seed=seed
        )
        prefix = f"gaussian_rho{rho_value}_seed{seed}"
    else:
        model = SACHistoryLogger(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            batch_size=256,
            verbose=0,
            device="cuda",
            seed=seed
        )
        prefix = f"sac_baseline_seed{seed}"
        
    callback = MetricsCallback(
        model_ref=model,
        eval_env=eval_env,
        eval_freq=625,
        n_eval_episodes=5,
        deterministic=True,
    )
    
    # 学習の実行（300万ステップ）
    model.learn(total_timesteps=total_steps, callback=callback)
    
    # --- 各種データの保存 ---
    np.savez(f"{SAVE_DIR}/{prefix}.npz", returns=np.array(callback.episode_returns), timesteps=np.array(callback.timesteps))
    np.savez(f"{SAVE_DIR}/{prefix}_entropy.npz", entropy=np.array(callback.entropies), timesteps=np.array(callback.timesteps))
    np.savez(f"{SAVE_DIR}/{prefix}_loss.npz", actor_loss=np.array(callback.actor_losses), critic_loss=np.array(callback.critic_losses), timesteps=np.array(callback.timesteps))
    
    # 後からのモデル重みの保存
    model.save(f"{SAVE_DIR}/{prefix}_model.zip")
    print(f"💾 半チーター用モデルとログの保存が完了しました: {prefix}")
    
    train_env.close()
    eval_env.close()

def main():
    seeds = [0] 
    
    for seed in seeds:
        # 1. 標準 SAC Baseline の実行
        run_experiment("baseline", rho_value=None, seed=seed, total_steps=3_000_000)
        
        # 2. Gaussian Prior (rho = 1.0) の実行
        run_experiment("gaussian", rho_value=1.0, seed=seed, total_steps=3_000_000)

if __name__ == "__main__":
    main()
