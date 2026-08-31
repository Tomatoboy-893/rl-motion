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
        self.entropies = []  # エントロピー保持用リスト追加

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # EvalCallbackによって評価が行われたタイミングを判定
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.episode_returns.append(self.last_mean_reward)
                self.timesteps.append(self.num_timesteps)

                # --- ポリシーのエントロピー取得処理 ---
                with torch.no_grad():
                    # リプレイバッファからミニバッチ(256)をサンプルして観測を取得
                    replay_data = self.model.replay_buffer.sample(256)
                    obs_tensor = self.model.actor.obs_to_tensor(replay_data.observations)[0]
                    
                    # 現在のアクターからアクション分布パラメータを取得し、平均エントロピーを算出
                    dist_params = self.model.actor.get_action_dist_params(obs_tensor)
                    entropy = self.model.actor.action_dist.entropy(dist_params).mean().item()

                self.entropies.append(entropy)

        return result

def make_envs():
    # ADRコードと完全に同じ環境設定 (n_envs=8)
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

    # 5回ランのループ
    for i in range(NUM_SEEDS):
        print(f"\n--- Standard SAC Baseline Run {i+1}/{NUM_SEEDS} ---")
        train_env, eval_env = make_envs()

        callback = UnifiedReturnCallback(
            eval_env=eval_env,
            eval_freq=625,  # n_envs=8 なので 625 * 8 = 5,000ステップごとに評価
            n_eval_episodes=5,
            deterministic=True,
        )

        # 標準SAC（Stable-Baselines3 純正）のモデル構築
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

        # 2. エントロピーデータ保存 (ADR側の保存形式と一致)
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
