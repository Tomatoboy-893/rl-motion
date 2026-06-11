# sac_laplace_entropy_sweep.py

import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sac_adr_main import SACWithLaplacePrior


# ===============================
# setup
# ===============================
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

SAVE_DIR = "./npz_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# callback
# ===============================
class UnifiedReturnCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.timesteps = []

    def _on_step(self):
        super()._on_step()

        if self.last_mean_reward is not None:
            self.episode_returns.append(self.last_mean_reward)
            self.timesteps.append(self.num_timesteps)

        return True


# ===============================
# env
# ===============================
def make_envs(seed):

    train_env = make_vec_env(
        "HalfCheetah-v5",
        n_envs=1,
        seed=seed
    )

    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    return train_env, eval_env


# ===============================
# run single experiment
# ===============================
def run_experiment(scale, seed, total_timesteps):

    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = SACWithLaplacePrior(
        "MlpPolicy",
        train_env,
        beta_kl=0.01,
        beta_lr=1e-3,
        target_kl=1.0,
        prior_std=scale,
        verbose=0,
        seed=seed,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # entropy取得
    ent = np.array(getattr(model, "pi_entropies", []))

    train_env.close()
    eval_env.close()

    return ent


# ===============================
# sweep
# ===============================
def run_sweep(scales, total_timesteps=1_000_000, n_seeds=5):

    all_means = []
    all_stds = []

    for scale in scales:

        print("\n================================")
        print(f"Laplace scale = {scale}")
        print("================================")

        entropies_all = []

        for seed in range(n_seeds):

            ent = run_experiment(scale, seed, total_timesteps)

            if len(ent) > 0:
                entropies_all.append(ent)

        # 長さ揃え
        min_len = min(len(x) for x in entropies_all)
        entropies_all = np.array([x[:min_len] for x in entropies_all])

        mean = entropies_all.mean(axis=0)
        std = entropies_all.std(axis=0)

        all_means.append(mean)
        all_stds.append(std)

        # 保存
        np.savez(
            f"{SAVE_DIR}/laplace_{scale}_entropy.npz",
            mean=mean,
            std=std,
            all_entropies=entropies_all,
        )

    return scales, all_means, all_stds


# ===============================
# plot
# ===============================
def plot_entropy(scales, means, stds):

    plt.figure(figsize=(7, 5))

    for scale, mean, std in zip(scales, means, stds):

        x = np.arange(len(mean))

        plt.plot(x, mean, label=f"scale={scale}")
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Training steps (evaluation index)")
    plt.ylabel("Policy entropy")
    plt.title("Laplace prior: entropy comparison")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/laplace_entropy_comparison.png",
        dpi=300
    )

    plt.close()


# ===============================
# main
# ===============================
def main():

    scales = [0.1, 0.5, 1.0, 2.0]

    scales, means, stds = run_sweep(scales)

    plot_entropy(scales, means, stds)

    print("\n[DONE] saved:")
    print(f"- {SAVE_DIR}/laplace_entropy_comparison.png")
    print(f"- {SAVE_DIR}/laplace_*_entropy.npz")


if __name__ == "__main__":
    main()
