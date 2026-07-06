import os
import time
import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# メインコードからガウス版クラスをインポート
from sac_adr_main import SACWithFixedPrior

SAVE_DIR = "./npz_logs_humanoid"
os.makedirs(SAVE_DIR, exist_ok=True)

class UnifiedReturnCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.timesteps = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None:
            self.episode_returns.append(self.last_mean_reward)
            self.timesteps.append(self.num_timesteps)
        return result

def make_envs():
    # 💡 n_envs=8 にして、8つの環境を同時にGPUに送り込みます
    train_env = make_vec_env("Humanoid-v5", n_envs=8, seed=None)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=None)
    return train_env, eval_env

def main():
    # 💡 実行したいスケールの全リスト
    SCALES = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    TOTAL_STEPS = 3_000_000  # 各300万ステップ
    NUM_SEEDS = 5            # 各5回実行

    start_time = time.time()
    print("=========================================")
    print(" Starting Humanoid-v5 FULL AUTO SWEEP")
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
            print(f"\n--- [scale={scale}] Run {i+1}/{NUM_SEEDS} ---")
            train_env, eval_env = make_envs()

            callback = UnifiedReturnCallback(
                eval_env=eval_env,
                eval_freq=20000,
                n_eval_episodes=5,
                deterministic=True,
            )

            # モデルのインスタンス化（論文準拠パラメータ）
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

            # データ保存
            prefix = f"gaussian_scale{scale}"
            
            # 1. 報酬データ保存
            np.savez(
                f"{SAVE_DIR}/{prefix}_run{i}.npz",
                returns=np.array(callback.episode_returns),
                timesteps=np.array(callback.timesteps),
            )
            
            # 2. エントロピーデータ保存
            if hasattr(model, "pi_entropies"):
                np.savez(
                    f"{SAVE_DIR}/{prefix}_run{i}_entropy.npz",
                    entropy=np.array(model.pi_entropies),
                )

            train_env.close()
            eval_env.close()
            print(f"[scale={scale}] Run {i+1} Done.")

    end_time = time.time()
    duration = (end_time - start_time) / 3600
    print(f"\n🎉 すべてのスケール（{SCALES}）の全自動実行が完了しました！")
    print(f"総所要時間: {duration:.2f} 時間")

if __name__ == "__main__":
    main()
