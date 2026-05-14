import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sac_adr_main import SACWithFixedPrior, SACWithLaplacePrior

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
        "HalfCheetah-v5",
        n_envs=1,
        seed=seed
    )

    eval_env = Monitor(gym.make("HalfCheetah-v5"))
    eval_env.reset(seed=seed + 1000)

    return train_env, eval_env

# ===============================
# Laplace prior 実験
# ===============================
def run_laplace(seed, total_timesteps, prior_std):

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

    prior_std=prior_std,

    verbose=1,
)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )

    # model save
    model.save(
        f"{SAVE_DIR}/laplace_b{prior_std}_seed{seed}_model"
    )

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    # seedごと保存
    np.savez(
        f"{SAVE_DIR}/laplace_b{prior_std}_seed{seed}.npz",
        timesteps=t,
        returns=r,
    )

    print(f"[Saved] laplace_b{prior_std}_seed{seed}.npz")

    return t, r

# ===============================
# メイン
# ===============================
def main():

    TOTAL_STEPS = 1_000_000

    # Laplace parameter sweep
    prior_std_list = [0.1, 0.5, 1.0, 2.0, 5.0]

    plt.figure(figsize=(7, 5))

    for prior_std in prior_std_list:

        print()
        print("====================================")
        print(f"Laplace b = {prior_std}")
        print("====================================")

        all_returns = []

        # 5 seeds
        for seed in range(5):

            print(f"Seed {seed}")

            t, r = run_laplace(
                seed=seed,
                total_timesteps=TOTAL_STEPS,
                prior_std=prior_std,
            )

            all_returns.append(r)

        # 途中でエピソードが終了した場合などに備え、配列の長さを揃える
        min_len = min(len(r) for r in all_returns)
        all_returns = np.array([r[:min_len] for r in all_returns])
        t = t[:min_len]

        mean = all_returns.mean(axis=0)
        std = all_returns.std(axis=0)

        # mean/std 保存
        np.savez(
            f"{SAVE_DIR}/laplace_b{prior_std}_mean_std.npz",
            timesteps=t,
            mean=mean,
            std=std,
            all_returns=all_returns,
        )

        print(f"[Saved] laplace_b{prior_std}_mean_std.npz")

        # plot
        plt.plot(
            t,
            mean,
            label=f"Laplace b={prior_std}"
        )

        plt.fill_between(
            t,
            mean - std,
            mean + std,
            alpha=0.2,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episode Return")
    plt.title("Laplace Prior Parameter Sweep")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        f"{SAVE_DIR}/laplace_parameter_sweep.png",
        dpi=300
    )

    plt.close()

    print()
    print("====================================")
    print("✅ laplace_parameter_sweep.png saved!")
    print("====================================")

if __name__ == "__main__":
    main()
