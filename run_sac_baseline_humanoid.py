import os
import time
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

SAVE_DIR = "./npz_logs_humanoid"
os.makedirs(SAVE_DIR, exist_ok=True)

class UnifiedReturnCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.timesteps = []
        self.entropies = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # 評価が行われるタイミング（n_envs=8でeval_freq=625の場合、5,000ステップごと）
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.episode_returns.append(self.last_mean_reward)
                self.timesteps.append(self.num_timesteps)

                # --- エントロピー取得処理（CUDA Tensorに対応） ---
                with torch.no_grad():
                    replay_data = self.model.replay_buffer.sample(256)
                    obs_tensor = replay_data.observations
                    
                    # アクション分布から log_prob を計算し、エントロピー (-log_prob) を求める
                    mean_actions, log_std, kwargs = self.model.actor.get_action_dist_params(obs_tensor)
                    _, log_prob = self.model.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)
                    entropy = (-log_prob).mean().item()

                self.entropies.append(entropy)

        return result

def make_envs():
    train_env = make_vec_env("Humanoid-v5", n_envs=8, seed=None)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=None)
    return train_env, eval_env

def main():
    TOTAL_STEPS = 3_000_000  # 300万ステップ
    NUM_SEEDS = 5            # 5シード

    start_time = time.time()
    print("=========================================")
    print(" Starting Humanoid-v5 Standard SAC BASELINE")
    print(f" Total Runs: {NUM_SEEDS} runs")
    print("=========================================")

    for i in range(NUM_SEEDS):
        print(f"\n--- Standard SAC Baseline Run {i+1}/{NUM_SEEDS} ---")
        train_env, eval_env = make_envs()

        callback = UnifiedReturnCallback(
            eval_env=eval_env,
            eval_freq=625,  # 625 * 8envs = 5,000ステップごとに評価
            n_eval_episodes=5,
            deterministic=True,
        )

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            batch_size=256,
            verbose=0,
            device="cuda"
        )

        # 学習開始
        model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

        prefix = "sac_baseline"

        # 1. 報酬データ保存
        np.savez(
            f"{SAVE_DIR}/{prefix}_run{i}.npz",
            returns=np.array(callback.episode_returns),
            timesteps=np.array(callback.timesteps),
        )

        # 2. エントロピーデータ保存
        np.savez(
            f"{SAVE_DIR}/{prefix}_run{i}_entropy.npz",
            entropy=np.array(callback.entropies),
            timesteps=np.array(callback.timesteps),
        )

        train_env.close()
        eval_env.close()
        print(f"Baseline Run {i+1} Done. Saved returns & entropy.")

    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print(f"\n🎉 標準SACベースライン（{NUM_SEEDS} runs）の実行が完了しました！")
    print(f"総所要時間: {duration:.2f} 時間")

if __name__ == "__main__":
    main()
