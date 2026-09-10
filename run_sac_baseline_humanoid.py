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
        
        # logger から毎ステップのロスを拾って一時保存
        logger_vals = self.model.logger.name_to_value
        if "train/actor_loss" in logger_vals and not np.isnan(logger_vals["train/actor_loss"]):
            self._temp_actor_losses.append(logger_vals["train/actor_loss"])
        if "train/critic_loss" in logger_vals and not np.isnan(logger_vals["train/critic_loss"]):
            self._temp_critic_losses.append(logger_vals["train/critic_loss"])
        
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

                # --- 2. この区間のロスの平均を記録 ---
                avg_actor_loss = np.mean(self._temp_actor_losses) if self._temp_actor_losses else np.nan
                avg_critic_loss = np.mean(self._temp_critic_losses) if self._temp_critic_losses else np.nan
                
                self.actor_losses.append(avg_actor_loss)
                self.critic_losses.append(avg_critic_loss)
                
                # 一時バッファをリセット
                self._temp_actor_losses = []
                self._temp_critic_losses = []

        return result

def main():
    # 動作確認のためまずは短めのステップ（例: 50,000ステップ）でテスト
    # 本番同様に回す場合は 3_000_000 にしてください
    TOTAL_STEPS = 3_000_000 
    
    print("=========================================")
    print(" Re-checking Loss Tracking (Test Run)")
    print("=========================================")

    train_env = make_vec_env("Humanoid-v5", n_envs=8, seed=42)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=42)

    callback = FixedLossCallback(
        eval_env=eval_env,
        eval_freq=625,  # 5,000ステップごと
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

    # 保存
    np.savez(
        f"{SAVE_DIR}/sac_recheck_loss.npz",
        actor_loss=np.array(callback.actor_losses),
        critic_loss=np.array(callback.critic_losses),
        timesteps=np.array(callback.timesteps),
    )
    print("✅ ロスの再測定データが保存されました: sac_recheck_loss.npz")

    train_env.close()
    eval_env.close()

if __name__ == "__main__":
    main()
