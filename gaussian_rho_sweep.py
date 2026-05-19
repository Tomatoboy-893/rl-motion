# gaussian_rho_sweep.py

import os
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sac_adr_main import SACWithFixedPrior


SAVE_DIR = "./npz_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Gaussian rho sweep
RHO_LIST = [0.05, 0.1, 0.2, 0.5]

# random seeds
SEEDS = [0, 1, 2, 3, 4]

# training steps
TOTAL_TIMESTEPS = 1_000_000

# evaluation interval
EVAL_FREQ = 5000


class RewardLoggerCallback(EvalCallback):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.timesteps_log = []
        self.reward_log = []

    def _on_step(self) -> bool:

        result = super()._on_step()

        if self.n_calls % self.eval_freq == 0:

            self.timesteps_log.append(self.num_timesteps)
            self.reward_log.append(self.last_mean_reward)

        return result


def run_gaussian(rho, seed):

    env = Monitor(gym.make("Humanoid-v4"))
    eval_env = Monitor(gym.make("Humanoid-v4"))

    model = SACWithFixedPrior(
        "MlpPolicy",
        env,

        verbose=1,
        seed=seed,

        beta_kl=1.0,
        beta_lr=1e-4,
        target_kl=0.01,

        prior_std=rho,

        device="cuda",
    )

    eval_callback = RewardLoggerCallback(
        eval_env,

        best_model_save_path=f"{SAVE_DIR}/gaussian_rho{rho}_seed{seed}_model",

        log_path=SAVE_DIR,

        eval_freq=EVAL_FREQ,

        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
    )

    timesteps = np.array(eval_callback.timesteps_log)
    rewards = np.array(eval_callback.reward_log)

    np.savez(
        f"{SAVE_DIR}/gaussian_rho{rho}_seed{seed}.npz",
        timesteps=timesteps,
        rewards=rewards,
    )

    print(f"[Saved] gaussian_rho{rho}_seed{seed}.npz")

    return timesteps, rewards


def main():

    for rho in RHO_LIST:

        print("\n====================================")
        print(f"Gaussian rho = {rho}")
        print("====================================")

        all_rewards = []

        for seed in SEEDS:

            print(f"Seed {seed}")

            t, r = run_gaussian(rho, seed)

            all_rewards.append(r)

        all_rewards = np.array(all_rewards)

        mean_rewards = np.mean(all_rewards, axis=0)
        std_rewards = np.std(all_rewards, axis=0)

        np.savez(
            f"{SAVE_DIR}/gaussian_rho{rho}_mean_std.npz",
            timesteps=t,
            mean=mean_rewards,
            std=std_rewards,
        )

        print(f"[Saved] gaussian_rho{rho}_mean_std.npz")

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
