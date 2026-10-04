# sac_rho_run.py (Comparison: SAC vs ADR (std=1.0) vs ADR (std=2.0))
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sac_adr_main import SACWithFixedPrior

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

SAVE_DIR = "./npz_logs_full_comparison"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 共通評価コールバック
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
    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed+1000)
    return train_env, eval_env

# ===============================
# 単一シードの実験実行関数
# ===============================
def run_experiment(algo_key, seed, total_timesteps):
    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    if algo_key == "sac":
        model = SAC("MlpPolicy", train_env, verbose=0)
    elif algo_key == "adr_std1.0":
        model = SACWithFixedPrior(
            "MlpPolicy", train_env,
            beta_kl=0.01, beta_lr=1e-3, target_kl=1.0,
            prior_std=1.0, verbose=0
        )
    elif algo_key == "adr_std2.0":
        model = SACWithFixedPrior(
            "MlpPolicy", train_env,
            beta_kl=0.01, beta_lr=1e-3, target_kl=1.0,
            prior_std=2.0, verbose=0
        )
    else:
        raise ValueError(f"Unknown algo_key: {algo_key}")

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # モデルの保存
    model.save(f"{SAVE_DIR}/{algo_key}_seed{seed}_model")

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    np.savez(f"{SAVE_DIR}/{algo_key}_seed{seed}.npz", timesteps=t, returns=r)
    print(f"[Saved] {SAVE_DIR}/{algo_key}_seed{seed}.npz & model")

    return t, r, model

# ===============================
# 5シード実行 + 平均・標準偏差の計算
# ===============================
def run_5seeds(algo_key, total_timesteps):
    all_returns = []
    models = []
    t = None

    for seed in range(5):
        print(f"[{algo_key.upper()}] Seed {seed} running...")
        t, r, model = run_experiment(algo_key, seed, total_timesteps)
        all_returns.append(r)
        models.append(model)

    all_returns = np.array(all_returns)
    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    np.savez(
        f"{SAVE_DIR}/{algo_key}_5seed_mean_std.npz",
        timesteps=t, mean=mean, std=std, all_returns=all_returns
    )
    return t, mean, std, models

# ===============================
# メイン処理（比較実行）
# ===============================
def main():
    TOTAL_STEPS = 3_000_000

    # 3つの条件で実行
    print("===== 1. Running Standard SAC =====")
    t_sac, mean_sac, std_sac, models_sac = run_5seeds("sac", TOTAL_STEPS)

    print("===== 2. Running ADR (prior_std=1.0) =====")
    t_adr1, mean_adr1, std_adr1, models_adr1 = run_5seeds("adr_std1.0", TOTAL_STEPS)

    print("===== 3. Running ADR (prior_std=2.0) =====")
    t_adr2, mean_adr2, std_adr2, models_adr2 = run_5seeds("adr_std2.0", TOTAL_STEPS)

    # ===============================
    # 1. リターンの比較グラフ
    # ===============================
    plt.figure(figsize=(9, 6))
    plt.plot(t_sac, mean_sac, label="Standard SAC", color="tab:blue")
    plt.fill_between(t_sac, mean_sac - std_sac, mean_sac + std_sac, color="tab:blue", alpha=0.2)

    plt.plot(t_adr1, mean_adr1, label="ADR (std=1.0)", color="tab:orange")
    plt.fill_between(t_adr1, mean_adr1 - std_adr1, mean_adr1 + std_adr1, color="tab:orange", alpha=0.2)

    plt.plot(t_adr2, mean_adr2, label="ADR (std=2.0)", color="tab:green")
    plt.fill_between(t_adr2, mean_adr2 - std_adr2, mean_adr2 + std_adr2, color="tab:green", alpha=0.2)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episodic Return")
    plt.title("Comparison: Return (SAC vs ADR std=1.0 vs ADR std=2.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/comparison_returns.png")
    plt.close()

    # ===============================
    # 2. エントロピーの比較グラフ（ADR 2種）
    # ===============================
    m1 = models_adr1[0]
    m2 = models_adr2[0]

    plt.figure(figsize=(8, 5))
    if hasattr(m1, "pi_entropies") and len(m1.pi_entropies) > 0:
        plt.plot(m1.pi_entropies, label="ADR (std=1.0)", color="tab:orange")
    if hasattr(m2, "pi_entropies") and len(m2.pi_entropies) > 0:
        plt.plot(m2.pi_entropies, label="ADR (std=2.0)", color="tab:green")
    plt.title("Comparison: Pi Entropy over updates")
    plt.xlabel("Updates")
    plt.ylabel("Entropy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/comparison_entropy.png")
    plt.close()

    # ===============================
    # 3. ロス / KLの比較グラフ（ADR 2種）
    # ===============================
    plt.figure(figsize=(8, 5))
    if hasattr(m1, "kl_values") and len(m1.kl_values) > 0:
        plt.plot(m1.kl_values, label="ADR (std=1.0) KL", color="tab:orange")
    if hasattr(m2, "kl_values") and len(m2.kl_values) > 0:
        plt.plot(m2.kl_values, label="ADR (std=2.0) KL", color="tab:green")
    plt.title("Comparison: KL(π||ρ) over updates")
    plt.xlabel("Updates")
    plt.ylabel("KL Divergence")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/comparison_kl_loss.png")
    plt.close()

    print(f"[Saved] All comparison graphs and models to {SAVE_DIR}/")

if __name__ == "__main__":
    main()
