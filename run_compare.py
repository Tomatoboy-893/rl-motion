# 比較実験実行用スクリプト
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from sac_adr_main import SACWithFixedPrior, SACWithLaplacePrior


# ===============================
# 保存先だお
# ===============================
SAVE_DIR = "./npz_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ===============================
# 評価コールバック
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
# 環境生成（重要：Monitor付き）
# ===============================
def make_envs(seed):
    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)

    # ★ここが重要修正
    eval_env = Monitor(gym.make("HalfCheetah-v5"))
    eval_env.reset(seed=seed + 1000)

    return train_env, eval_env


# ===============================
# 単発実行
# ===============================
def run_single(model_class, name, seed, total_timesteps):
    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model = model_class(
        "MlpPolicy",
        train_env,
        beta_kl=0.01,
        beta_lr=1e-3,
        target_kl=1.0,
        prior_std=2.0,
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    np.savez(
        f"{SAVE_DIR}/{name}_seed{seed}.npz",
        timesteps=t,
        returns=r
    )

    train_env.close()
    eval_env.close()

    return t, r


# ===============================
# 5 seed 実行
# ===============================
def run_experiment(model_class, name, total_timesteps):
    all_returns = []

    for seed in range(5):
        print(f"{name} Seed {seed}")
        t, r = run_single(model_class, name, seed, total_timesteps)
        all_returns.append(r)

    all_returns = np.array(all_returns)

    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    np.savez(
        f"{SAVE_DIR}/{name}_5seed_mean_std.npz",
        timesteps=t,
        mean=mean,
        std=std,
        all_returns=all_returns
    )

    return t, mean, std


# ===============================
# 比較プロット
# ===============================
def plot_compare(t, m1, s1, m2, s2):
    plt.figure(figsize=(7,5))

    plt.plot(t, m1, label="Gaussian")
    plt.fill_between(t, m1 - s1, m1 + s1, alpha=0.2)

    plt.plot(t, m2, label="Laplace")
    plt.fill_between(t, m2 - s2, m2 + s2, alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episodic Return")
    plt.legend()
    plt.grid()

    plt.savefig(f"{SAVE_DIR}/comparison.png", dpi=300)
    plt.close()


# ===============================
# メイン
# ===============================
def main():
    TOTAL_STEPS = 1_000_000   # ← 3_000_000に変えてOK

    print("===== Gaussian =====")
    t_g, m_g, s_g = run_experiment(
        SACWithFixedPrior, "gaussian", TOTAL_STEPS
    )

    print("===== Laplace =====")
    t_l, m_l, s_l = run_experiment(
        SACWithLaplacePrior, "laplace", TOTAL_STEPS
    )

    # 長さ揃え
    min_len = min(len(t_g), len(t_l))
    t = t_g[:min_len]

    plot_compare(
        t,
        m_g[:min_len], s_g[:min_len],
        m_l[:min_len], s_l[:min_len]
    )

    print("✅ comparison.png saved!")


if __name__ == "__main__":
    main()
