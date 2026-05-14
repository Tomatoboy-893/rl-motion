# 単一モデル実行用スクリプト
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# 👇 ここだけ変更
from sac_adr_main import SACWithLaplacePrior

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

SAVE_DIR = "./npz_logs"
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


def make_envs(seed):
    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed+1000)
    return train_env, eval_env


def run_sac(seed, total_timesteps):
    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    model = SACWithLaplacePrior(
        "MlpPolicy",
        train_env,
        beta_kl=0.01,
        beta_lr=1e-3,
        target_kl=1.0,
        prior_std=2.0,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    return t, r, model


def main():
    TOTAL_STEPS = 1_000_000

    all_returns = []
    for seed in range(5):
        t, r, _ = run_sac(seed, TOTAL_STEPS)
        all_returns.append(r)

    min_len = min(len(r) for r in all_returns)
    all_returns = np.array([r[:min_len] for r in all_returns])
    t = t[:min_len]

    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    plt.plot(t, mean, label="Laplace Prior")
    plt.fill_between(t, mean-std, mean+std, alpha=0.3)
    plt.legend()
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episodic Return")
    plt.grid()
    plt.savefig("learning_curve_laplace.png")


if __name__ == "__main__":
    main()
