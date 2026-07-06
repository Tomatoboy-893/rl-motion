import os
import argparse
import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# あなたのメインコードからインポート
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
    train_env = make_vec_env("Humanoid-v5", n_envs=1, seed=None)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=None)
    return train_env, eval_env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, required=True, choices=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    args = parser.parse_args()

    TOTAL_STEPS = 3_000_000  
    NUM_SEEDS = 5            

    print(f"=========================================")
    print(f" Starting Humanoid-v5 Experiment (Gaussian)")
    print(f" Scale: {args.scale} | Total Steps: {TOTAL_STEPS}")
    print(f" Seed Mode: Non-fixed (Aoyagi Style)")
    print(f"=========================================")

    for i in range(NUM_SEEDS):
        print(f"\n--- Running Run {i+1}/{NUM_SEEDS} ---")
        train_env, eval_env = make_envs()

        callback = UnifiedReturnCallback(
            eval_env=eval_env,
            eval_freq=20000,
            n_eval_episodes=5,
            deterministic=True,
        )

        # 💡 mainの実装はいじらず、インスタンス化の引数で論文の標準パラメータを叩き込みます
        model = SACWithFixedPrior(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,  # 論文標準の学習率
            batch_size=256,      # 論文標準のバッチサイズ（Humanoidの広大な空間を安定させる）
            beta_kl=0.01,
            beta_lr=1e-3,
            target_kl=1.0,
            prior_std=args.scale,
            verbose=0,
            device="cuda"
        )

        model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

        prefix = f"gaussian_scale{args.scale}"
        
        np.savez(
            f"{SAVE_DIR}/{prefix}_run{i}.npz",
            returns=np.array(callback.episode_returns),
            timesteps=np.array(callback.timesteps),
        )
        
        if hasattr(model, "pi_entropies"):
            np.savez(
                f"{SAVE_DIR}/{prefix}_run{i}_entropy.npz",
                entropy=np.array(model.pi_entropies),
            )

        train_env.close()
        eval_env.close()
        print(f"Run {i+1} Done.")

    print(f"\n🎉 Gaussian scale={args.scale} の5ランがすべて完了しました！")

if __name__ == "__main__":
    main()
