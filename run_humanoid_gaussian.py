import os
import time
import numpy as np
import torch
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# メインコードからガウス版クラスをインポート
from sac_adr_main import SACWithFixedPrior

SAVE_DIR = "./npz_logs_humanoid"
os.makedirs(SAVE_DIR, exist_ok=True)

class LossAndReturnCallback(EvalCallback):
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

                # --- 1. エントロピー取得（モデル側で保持していればそれを使うか、安全に計算） ---
                with torch.no_grad():
                    try:
                        replay_data = self.model.replay_buffer.sample(256)
                        obs_tensor = replay_data.observations
                        mean_actions, log_std, kwargs_dist = self.model.actor.get_action_dist_params(obs_tensor)
                        _, log_prob = self.model.actor.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs_dist)
                        entropy = (-log_prob).mean().item()
                    except Exception:
                        entropy = 0.0
                self.entropies.append(entropy)

                # --- 2. この区間のロスの平均を記録（データがなければ直前の値や0でフォールバック） ---
                if self._temp_actor_losses:
                    avg_actor_loss = float(np.mean(self._temp_actor_losses))
                    self._last_actor_loss = avg_actor_loss
                else:
                    avg_actor_loss = getattr(self, '_last_actor_loss', 0.0)

                if self._temp_critic_losses:
                    avg_critic_loss = float(np.mean(self._temp_critic_losses))
                    self._last_critic_loss = avg_critic_loss
                else:
                    avg_critic_loss = getattr(self, '_last_critic_loss', 0.0)
                
                self.actor_losses.append(avg_actor_loss)
                self.critic_losses.append(avg_critic_loss)
                
                # 一時バッファをリセット
                self._temp_actor_losses = []
                self._temp_critic_losses = []

        return result

def make_envs():
    # n_envs=8 にして、8つの環境を同時に並列実行
    train_env = make_vec_env("Humanoid-v5", n_envs=8, seed=None)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=None)
    return train_env, eval_env

def main():
    SCALES = [0.5, 1.0, 5.0]
    TOTAL_STEPS = 3_000_000  # 各300万ステップ
    NUM_SEEDS = 5            # 各5回実行（seed0〜seed4）

    start_time = time.time()
    print("=========================================")
    print(" Starting Humanoid-v5 FULL AUTO SWEEP (With Loss)")
    print(f" Targets: {SCALES}")
    print(f" Total Runs: {len(SCALES) * NUM_SEEDS} runs")
    print("=========================================")

    # スケールのループ
    for scale in SCALES:
        print(f"\n#########################################")
        print(f"  STARTING SWEEP: scale = {scale}")
        print(f"#########################################")
        
        # 5回ランのループ
        for i in range(NUM_SEEDS):
            print(f"\n--- [scale={scale}] Run {i+1}/{NUM_SEEDS} (Seed {i}) ---")
            train_env, eval_env = make_envs()

            callback = LossAndReturnCallback(
                eval_env=eval_env,
                eval_freq=625,
                n_eval_episodes=5,
                deterministic=True,
            )

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

            # 学習開始
            model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

            # 保存用プレフィックス
            prefix = f"gaussian_scale{scale}_seed{i}"
            
            # 1. 報酬データ保存 (.npz)
            np.savez(
                f"{SAVE_DIR}/{prefix}.npz",
                returns=np.array(callback.episode_returns),
                timesteps=np.array(callback.timesteps),
            )
            
            # 2. エントロピーデータ保存 (.npz)
            if hasattr(model, "pi_entropies") and model.pi_entropies:
                np.savez(
                    f"{SAVE_DIR}/{prefix}_entropy.npz",
                    entropy=np.array(model.pi_entropies),
                    timesteps=np.array(callback.timesteps),
                )
            else:
                np.savez(
                    f"{SAVE_DIR}/{prefix}_entropy.npz",
                    entropy=np.array(callback.entropies),
                    timesteps=np.array(callback.timesteps),
                )

            # 3. 💡 ロスデータ保存 (.npz) [新規追加]
            np.savez(
                f"{SAVE_DIR}/{prefix}_loss.npz",
                actor_loss=np.array(callback.actor_losses),
                critic_loss=np.array(callback.critic_losses),
                timesteps=np.array(callback.timesteps),
            )

            # 4. モデルの重み（.zipファイル）を保存
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
