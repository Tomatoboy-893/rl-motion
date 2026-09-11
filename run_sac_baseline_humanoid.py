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

class FixedLossCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.timesteps = []
        self.entropies = []
        self.actor_losses = []
        self.critic_losses = []
        
        # 一時保存用バッファ
        self._temp_actor_losses = []
        self._temp_critic_losses = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # logger から毎ステップのロスを拾って一時保存（nanは除外）
        logger_vals = self.model.logger.name_to_value
        if "train/actor_loss" in logger_vals:
            val = logger_vals["train/actor_loss"]
            if val is not None and not np.isnan(val):
                self._temp_actor_losses.append(val)
                
        if "train/critic_loss" in logger_vals:
            val = logger_vals["train/critic_loss"]
            if val is not None and not np.isnan(val):
                self._temp_critic_losses.append(val)
        
        # 評価が行われるタイミング（eval_freqごと）
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.episode_returns.append(self.last_mean_reward)
                self.timesteps.append(self.num_timesteps)

                # --- 1. エントロピー取得 ---
                with torch.no_grad():
                    replay_data = self.model.replay_buffer.sample(256)
                    obs_tensor = replay_data.observations
                    mean_actions, log_std, kwargs_dist = self.model.actor.get_action_dist_params(obs_tensor)
                    _, log_prob = self.model.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs_dist)
                    entropy = (-log_prob).mean().item()
                self.entropies.append(entropy)

                # --- 2. この区間のロスの平均を記録（データがなければ直前の値や0でフォールバック） ---
                if self._temp_actor_losses:
                    avg_actor_loss = np.mean(self._temp_actor_losses)
                else:
                    avg_actor_loss = self.actor_losses[-1] if self.actor_losses else 0.0

                if self._temp_critic_losses:
                    avg_critic_loss = np.mean(self._temp_critic_losses)
                else:
                    avg_critic_loss = self.critic_losses[-1] if self.critic_losses else 0.0
                
                self.actor_losses.append(avg_actor_loss)
                self.critic_losses.append(avg_critic_loss)
                
                # 一時バッファをリセット
                self._temp_actor_losses = []
                self._temp_critic_losses = []

        return result

def main():
    TOTAL_STEPS = 3_000_000  # 本番用の300万ステップ
    NUM_SEEDS = 5            # 5シード分回す場合

    print("=========================================")
    print(" Starting Humanoid-v5 Loss-Fixed Training")
    print("=========================================")

    for i in range(NUM_SEEDS):
        print(f"\n--- Run {i+1}/{NUM_SEEDS} ---")
        train_env = make_vec_env("Humanoid-v5", n_envs=8, seed=None)
        eval_env = gym.make("Humanoid-v5")
        eval_env.reset(seed=None)

        callback = FixedLossCallback(
            eval_env=eval_env,
            eval_freq=625,  # 8環境で5,000ステップごと
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

        model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

        # 保存（従来のファイル名と一致させることで既存のプロットスクリプトがそのまま使える）
        np.savez(
            f"{SAVE_DIR}/sac_baseline_run{i}.npz",
            returns=np.array(callback.episode_returns),
            timesteps=np.array(callback.timesteps),
        )
        np.savez(
            f"{SAVE_DIR}/sac_baseline_run{i}_entropy.npz",
            entropy=np.array(callback.entropies),
            timesteps=np.array(callback.timesteps),
        )
        np.savez(
            f"{SAVE_DIR}/sac_baseline_run{i}_loss.npz",
            actor_loss=np.array(callback.actor_losses),
            critic_loss=np.array(callback.critic_losses),
            timesteps=np.array(callback.timesteps),
        )

        train_env.close()
        eval_env.close()
        print(f"✅ Run {i+1} Done & Saved.")

    print("\n🎉 すべての処理が完了しました！")

if __name__ == "__main__":
    main()
