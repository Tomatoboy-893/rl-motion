# sac_rho_run.py (Moving Average Prior - Single Method 5-Seed Run & Metrics Export)
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sac_adr_main import SACWithMovingAveragePrior

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

SAVE_DIR = "./npz_logs_moving_average_prior"
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
# 単一シードの実験実行関数（移動平均プライオ専用）
# ===============================
def run_experiment(seed, total_timesteps):
    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model = SACWithMovingAveragePrior(
        "MlpPolicy", train_env,
        beta_kl=0.01, beta_lr=1e-3, target_kl=1.0,
        prior_std=1.0, tau=0.005, use_value_filter=True, verbose=0
    )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)

    # モデルの保存
    model.save(f"{SAVE_DIR}/adr_moving_avg_seed{seed}_model")

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    np.savez(f"{SAVE_DIR}/adr_moving_avg_seed{seed}.npz", timesteps=t, returns=r)
    print(f"[Saved] {SAVE_DIR}/adr_moving_avg_seed{seed}.npz & model")

    return t, r, model

# ===============================
# メイン処理（5シード実行と4指標の可視化）
# ===============================
def main():
    TOTAL_STEPS = 3_000_000
    all_returns = []
    models = []
    t = None

    print("===== Running ADR (Moving Average Prior) - 5 Seeds =====")
    for seed in range(5):
        print(f"--- Seed {seed} running ---")
        t, r, model = run_experiment(seed, TOTAL_STEPS)
        all_returns.append(r)
        models.append(model)

    all_returns = np.array(all_returns)
    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    # 5シード平均・標準偏差の保存
    np.savez(
        f"{SAVE_DIR}/adr_moving_avg_5seed_mean_std.npz",
        timesteps=t, mean=mean, std=std, all_returns=all_returns
    )

    # ===============================
    # 1. 学習曲線（Mean Episodic Return ± std）
    # ===============================
    plt.figure(figsize=(8, 5))
    plt.plot(t, mean, label="ADR (Moving Avg Prior)", color="tab:purple", linewidth=2)
    plt.fill_between(t, mean - std, mean + std, color="tab:purple", alpha=0.2)
    plt.xlabel("Timesteps", fontsize=11)
    plt.ylabel("Mean Episodic Return", fontsize=11)
    plt.title("Learning Curve: ADR (Moving Average Prior)", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/moving_avg_returns.png", dpi=300)
    plt.close()

    # ===============================
    # 2. プライオ平均のノルム軌跡（seed 0）
    # ===============================
    m_ma = models[0]
    if hasattr(m_ma, "prior_mean_logs") and len(m_ma.prior_mean_logs) > 0:
        plt.figure(figsize=(8, 5))
        mean_norms = [np.linalg.norm(m) for m in m_ma.prior_mean_logs]
        plt.plot(mean_norms, label="Norm of Dynamic Prior Mean ||μ_ρ||", color="tab:purple", linewidth=1.5)
        plt.title("Evolution of Moving Average Prior Mean Norm", fontsize=13)
        plt.xlabel("Updates", fontsize=11)
        plt.ylabel("L2 Norm", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/moving_avg_prior_mean_norm.png", dpi=300)
        plt.close()

    # ===============================
    # 3. KLダイバージェンスの推移（seed 0）
    # ===============================
    if hasattr(m_ma, "kl_values") and len(m_ma.kl_values) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(m_ma.kl_values, label="KL(π||ρ)", color="tab:purple", alpha=0.8)
        plt.title("KL(π||ρ) over updates (Moving Average)", fontsize=13)
        plt.xlabel("Updates", fontsize=11)
        plt.ylabel("KL Divergence", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/moving_avg_kl.png", dpi=300)
        plt.close()

    # ===============================
    # 4. 方策エントロピーの推移（seed 0）
    # ===============================
    if hasattr(m_ma, "pi_entropies") and len(m_ma.pi_entropies) > 0:
        plt.figure(figsize=(8, 5))
        plt.plot(m_ma.pi_entropies, label="Pi Entropy", color="tab:purple", alpha=0.8)
        plt.title("Policy Entropy over updates (Moving Average)", fontsize=13)
        plt.xlabel("Updates", fontsize=11)
        plt.ylabel("Entropy", fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/moving_avg_entropy.png", dpi=300)
        plt.close()

    print(f"[Saved] All metrics and graphs for Moving Average Prior to {SAVE_DIR}/")

if __name__ == "__main__":
    main()
