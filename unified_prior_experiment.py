import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sac_adr_main import (
    SACWithFixedPrior,
    SACWithLaplacePrior
)

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
# callback
# ===============================
class UnifiedReturnCallback(EvalCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.episode_returns = []
        self.timesteps = []

    def _on_step(self):

        result = super()._on_step()

        if self.last_mean_reward is not None:

            self.episode_returns.append(
                self.last_mean_reward
            )

            self.timesteps.append(
                self.num_timesteps
            )

        return result

# ===============================
# env
# ===============================
def make_envs(seed):

    train_env = make_vec_env(
        "HalfCheetah-v5",
        n_envs=1,
        seed=seed
    )

    eval_env = Monitor(
        gym.make("HalfCheetah-v5")
    )

    eval_env.reset(seed=seed + 1000)

    return train_env, eval_env

# ===============================
# main experiment
# ===============================
def run_experiment(
    prior_type,
    prior_param,
    seed,
    total_timesteps,
):

    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # ===================================
    # model
    # ===================================
    if prior_type == "gaussian":

        model = SACWithFixedPrior(
            "MlpPolicy",
            train_env,

            beta_kl=0.01,
            beta_lr=1e-3,
            target_kl=1.0,

            prior_std=prior_param,

            verbose=0,
            device="cuda",
            seed=seed,
        )

    elif prior_type == "laplace":

        model = SACWithLaplacePrior(
            "MlpPolicy",
            train_env,

            beta_kl=0.01,
            beta_lr=1e-3,
            target_kl=1.0,

            prior_std=prior_param,

            verbose=0,
            device="cuda",
            seed=seed,
        )

    else:
        raise ValueError("invalid prior_type")

    # ===================================
    # learn
    # ===================================
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    # ===================================
    # save
    # ===================================
    prefix = f"{prior_type}_{prior_param}"

    model.save(
        f"{SAVE_DIR}/{prefix}_seed{seed}_model"
    )

    returns = np.array(callback.episode_returns)

    timesteps = np.array(callback.timesteps)

    entropies = np.array(model.pi_entropies)

    # return save
    np.savez(
        f"{SAVE_DIR}/{prefix}_seed{seed}.npz",
        timesteps=timesteps,
        returns=returns,
    )

    # entropy save
    np.savez(
        f"{SAVE_DIR}/{prefix}_seed{seed}_entropy.npz",
        entropy=entropies,
    )

    print(f"[Saved] {prefix}_seed{seed}")

    train_env.close()
    eval_env.close()

    return timesteps, returns, entropies

# ===============================
# sweep
# ===============================
def run_sweep(
    prior_type,
    param_list,
    total_timesteps,
):

    plt.figure(figsize=(7,5))

    for param in param_list:

        print()
        print("================================")
        print(f"{prior_type} param = {param}")
        print("================================")

        all_returns = []
        all_entropies = []

        for seed in range(5):

            t, r, e = run_experiment(
                prior_type=prior_type,
                prior_param=param,
                seed=seed,
                total_timesteps=total_timesteps,
            )

            all_returns.append(r)
            all_entropies.append(e)

        # ============================
        # return mean/std
        # ============================
        min_len = min(len(x) for x in all_returns)

        all_returns = np.array([
            x[:min_len] for x in all_returns
        ])

        t = t[:min_len]

        mean = all_returns.mean(axis=0)
        std = all_returns.std(axis=0)

        prefix = f"{prior_type}_{param}"

        np.savez(
            f"{SAVE_DIR}/{prefix}_mean_std.npz",
            timesteps=t,
            mean=mean,
            std=std,
            all_returns=all_returns,
        )

        # ============================
        # entropy mean/std
        # ============================
        e_min = min(len(x) for x in all_entropies)

        all_entropies = np.array([
            x[:e_min] for x in all_entropies
        ])

        entropy_mean = all_entropies.mean(axis=0)
        entropy_std = all_entropies.std(axis=0)

        np.savez(
            f"{SAVE_DIR}/{prefix}_entropy_mean_std.npz",
            mean=entropy_mean,
            std=entropy_std,
            all_entropies=all_entropies,
        )

        # ============================
        # plot
        # ============================
        plt.plot(
            t,
            mean,
            label=f"{prior_type}={param}"
        )

        plt.fill_between(
            t,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Return")

    plt.title(f"{prior_type} parameter sweep")

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/{prior_type}_sweep.png",
        dpi=300
    )

    plt.close()

# ===============================
# main
# ===============================

def main():

    TOTAL_STEPS = 1_000_000

    run_sweep(
        prior_type="laplace",
        param_list=[
            0.1,
            0.5,
            1.0,
            2.0,
        ],
        total_timesteps=TOTAL_STEPS,
    )

if __name__ == "__main__":
    main()


