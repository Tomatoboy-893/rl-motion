# gaussian_rho_sweep.py

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sac_adr_main import SACWithFixedPrior


# ===============================
# OpenGL
# ===============================
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# ===============================
# 保存先
# ===============================
SAVE_DIR = "./npz_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 評価callback
# ===============================
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


# ===============================
# 環境生成
# ===============================
def make_envs(seed):

    train_env = make_vec_env(
        "Humanoid-v4",
        n_envs=1,
        seed=seed
    )

    eval_env = Monitor(gym.make("Humanoid-v4"))
    eval_env.reset(seed=seed + 1000)

    return train_env, eval_env


# ===============================
# Gaussian prior 実験
# ===============================
def run_gaussian(seed, total_timesteps, rho):

    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = SACWithFixedPrior(
        "MlpPolicy",
        train_env,

        beta_kl=1.0,
        beta_lr=1e-4,
        target_kl=0.01,

        prior_std=rho,

        verbose=1,
        device="cuda",
        seed=seed,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    # model save
    model.save(
        f"{SAVE_DIR}/gaussian_rho{rho}_seed{seed}_model"
    )

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    # seedごと保存
    np.savez(
        f"{SAVE_DIR}/gaussian_rho{rho}_seed{seed}.npz",
        timesteps=t,
        returns=r,
    )

    print(f"[Saved] gaussian_rho{rho}_seed{seed}.npz")

    return t, r


# ===============================
# メイン
# ===============================
def main():

    TOTAL_STEPS = 1_000_000

    # Gaussian rho sweep
    rho_list = [0.05, 0.1, 0.2, 0.5]

    plt.figure(figsize=(7, 5))

    for rho in rho_list:

        print()
        print("====================================")
        print(f"Gaussian rho = {rho}")
        print("====================================")

        all_returns = []

        # 5 seeds
        for seed in range(5):

            print(f"Seed {seed}")

            t, r = run_gaussian(
                seed=seed,
                total_timesteps=TOTAL_STEPS,
                rho=rho,
            )

            all_returns.append(r)

        # 長さ合わせ
        min_len = min(len(r) for r in all_returns)

        all_returns = np.array([
            r[:min_len] for r in all_returns
        ])

        t = t[:min_len]

        mean = all_returns.mean(axis=0)
        std = all_returns.std(axis=0)

        # mean/std 保存
        np.savez(
            f"{SAVE_DIR}/gaussian_rho{rho}_mean_std.npz",
            timesteps=t,
            mean=mean,
            std=std,
            all_returns=all_returns,
        )

        print(f"[Saved] gaussian_rho{rho}_mean_std.npz")

        # plot
        plt.plot(
            t,
            mean,
            label=f"Gaussian rho={rho}"
        )

        plt.fill_between(
            t,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Return")

    plt.title("Gaussian Prior Parameter Sweep")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/gaussian_parameter_sweep.png",
        dpi=300
    )

    plt.close()

    print()
    print("====================================")
    print("✅ gaussian_parameter_sweep.png saved!")
    print("====================================")


if __name__ == "__main__":
    main()
