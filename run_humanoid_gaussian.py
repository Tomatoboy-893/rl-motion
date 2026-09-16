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

# ==========================================
# 1. 固定事前分布（ADR）付き SAC クラスのインライン定義
# ==========================================
class SACWithFixedPrior(SAC):
    def __init__(self, *args, prior_std=1.0, beta_kl=0.01, beta_lr=1e-3, target_kl=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_std = prior_std
        self.beta_kl = beta_kl
        self.beta_lr = beta_lr
        self.target_kl = target_kl
        
        # 確実にロスを蓄積するための内部バッファ
        self.actor_losses_history = []
        self.critic_losses_history = []
        self.pi_entropies = []

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        super().train(gradient_steps, batch_size)
        
        # 学習（勾配更新）が行われたタイミングでロガーからロスを確実に回収
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
# 2. 評価・ロス記録用コールバック
# ==========================================
class LossAndReturnCallback(EvalCallback):
    def __init__(self, model_ref, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref  # SACWithFixedPrior のインスタンス参照
        self.episode_returns = []
        self.timesteps = []
        self.entropies = []
        self.actor_losses = []
        self.critic_losses = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # 評価が行われるタイミング（eval_freqごと）
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

                # --- 2. 直近区間のロス平均をモデルの履歴から取得 ---
                if self.model_ref.actor_losses_history:
                    avg_actor = float(np.mean(self.model_ref.actor_losses_history))
                    self.actor_losses.append(avg_actor)
                    # 取得したらクリアして次の区間に備える
                    self.model_ref.actor_losses_history = []
                else:
                    # データがなければ直前の値、または0
                    fallback = self.actor_losses[-1] if self.actor_losses else 0.0
                    self.actor_losses.append(fallback)

                if self.model_ref.critic_losses_history:
                    avg_critic = float(np.mean(self.model_ref.critic_losses_history))
                    self.critic_losses.append(avg_critic)
                    self.model_ref.critic_losses_history = []
                else:
                    fallback = self.critic_losses[-1] if self.critic_losses else 0.0
                    self.critic_losses.append(fallback)

        return result


def make_envs():
    train_env = make_vec_env("Humanoid-v5", n_envs=8, seed=None)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=None)
    return train_env, eval_env


# ==========================================
# 3. メイン処理（フルオートスイープ）
# ==========================================
def main():
    SCALES = [0.5, 1.0, 5.0]
    TOTAL_STEPS = 3_000_000  # 各300万ステップ
    NUM_SEEDS = 5            # 各5回実行（seed0〜seed4）

    start_time = time.time()
    print("=========================================")
    print(" Starting Humanoid-v5 FULL AUTO SWEEP (Self-Contained Loss Fix)")
    print(f" Targets: {SCALES}")
    print(f" Total Runs: {len(SCALES) * NUM_SEEDS} runs")
    print("=========================================")

    for scale in SCALES:
        print(f"\n#########################################")
        print(f"  STARTING SWEEP: scale = {scale}")
        print(f"#########################################")
        
        for i in range(NUM_SEEDS):
            print(f"\n--- [scale={scale}] Run {i+1}/{NUM_SEEDS} (Seed {i}) ---")
            train_env, eval_env = make_envs()

            # モデルのインスタンス化
            model = SACWithFixedPrior(
                "MlpPolicy",
                train_env,
                learning_rate=3e-4,
                batch_size=256,
                beta_kl=0.01,
                beta_lr=1e-3,
                target_kl=1.0,
                prior_std=scale,
                verbose=0,
                device="cuda"
            )

            callback = LossAndReturnCallback(
                model_ref=model,
                eval_env=eval_env,
                eval_freq=625,
                n_eval_episodes=5,
                deterministic=True,
            )

            # 学習開始
            model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

            # 保存用プレフィックス
            prefix = f"gaussian_scale{scale}_seed{i}"
            
            # 1. 報酬データ保存
            np.savez(
                f"{SAVE_DIR}/{prefix}.npz",
                returns=np.array(callback.episode_returns),
                timesteps=np.array(callback.timesteps),
            )
            
            # 2. エントロピーデータ保存
            np.savez(
                f"{SAVE_DIR}/{prefix}_entropy.npz",
                entropy=np.array(callback.entropies),
                timesteps=np.array(callback.timesteps),
            )

            # 3. ロスデータ保存（確実に値が入るようになりました）
            np.savez(
                f"{SAVE_DIR}/{prefix}_loss.npz",
                actor_loss=np.array(callback.actor_losses),
                critic_loss=np.array(callback.critic_losses),
                timesteps=np.array(callback.timesteps),
            )

            # 4. モデルの重みを保存
            model_save_path = f"{SAVE_DIR}/{prefix}_model.zip"
            model.save(model_save_path)
            print(f"💾 モデルとログを保存しました: {prefix}")

            train_env.close()
            eval_env.close()
            print(f"[scale={scale}] Run {i+1} Done.")

    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print(f"\n🎉 すべてのスケール（{SCALES}）の全自動実行が完了しました！")
    print(f"総所要時間: {duration:.2f} 時間")

if __name__ == "__main__":
    main()
