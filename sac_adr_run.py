# sac_rho_run.py
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sac_adr_main import SACWithFixedPrior

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

SAVE_DIR = "./npz_logs"
os.makedirs(SAVE_DIR, exist_ok=True)

# ===============================
# 共通評価コールバック（完全一致）
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
# 環境生成（完全共通）
# ===============================
def make_envs(seed):
    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed+1000)
    return train_env, eval_env

# ===============================
# SAC + ρ 実行（seed単体）
# ===============================
def run_sac_rho(seed, total_timesteps):
    train_env, eval_env = make_envs(seed)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    model = SACWithFixedPrior(
        "MlpPolicy",
        train_env,
        beta_kl=0.01,
        beta_lr=1e-3,
        target_kl=1.0,
        prior_std=2.0,
        verbose=1,
    )


    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    # save model + extra
    model.save(f"{SAVE_DIR}/sac_rho_seed{seed}_model")

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    # seedごと即npz保存
    np.savez(
        f"{SAVE_DIR}/sac_rho_seed{seed}.npz",
        timesteps=t,
        returns=r
    )
    print(f"[Saved] {SAVE_DIR}/sac_rho_seed{seed}.npz")

    return t, r, model

# ===============================
# 5 seed 実行 + mean ± std
# ===============================
def run(run_func, total_timesteps):
    all_returns = []
    models = []
    t = None

    for seed in range(5):
        print(f"Seed {seed} running...")
        t, r, model = run_func(seed, total_timesteps)
        all_returns.append(r)
        models.append(model)

    all_returns = np.array(all_returns)
    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    # mean ± std を保存（論文用）
    np.savez(
        f"{SAVE_DIR}/sac_rho_5seed_mean_std.npz",
        timesteps=t,
        mean=mean,
        std=std,
        all_returns=all_returns
    )
    print(f"[Saved] {SAVE_DIR}/sac_rho_5seed_mean_std.npz")

    return t, mean, std, models

# ===============================
# メイン処理
# ===============================
def main():
    TOTAL_STEPS = 3_000_000

    print("===== Running SAC + ρ =====")
    t_rho, r_rho, std_rho, models = run(run_sac_rho, TOTAL_STEPS)

    # 描画保存（学習曲線）
    plt.figure(figsize=(6, 5))
    plt.plot(t_rho, r_rho, label="SAC + rho mean")
    plt.fill_between(
        t_rho,
        r_rho - std_rho,
        r_rho + std_rho,
        alpha=0.3
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean Episodic Return")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/sac_rho_learning_curve.png")
    plt.close()

    # pick first model for per-update logs (if available)
    model = models[0] if len(models) > 0 else None

    # --- KL(π||ρ) ---
    if model is not None and hasattr(model, "kl_values") and len(model.kl_values) > 0:
        plt.figure(figsize=(6,5))
        plt.plot(model.kl_values)
        plt.title("KL(π||ρ) over training")
        plt.xlabel("Updates")
        plt.ylabel("KL")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/training_kl_pi_rho.png")
        plt.close()

    # --- π entropy ---
    if model is not None and hasattr(model, "pi_entropies") and len(model.pi_entropies) > 0:
        plt.figure(figsize=(6, 5))
        plt.plot(model.pi_entropies)
        plt.title("Pi Entropy over training")
        plt.xlabel("Updates")
        plt.ylabel("Entropy")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/training_pi_entropy.png")
        plt.close()

    # --- β logs ---
    if model is not None and hasattr(model, "beta_logs") and len(model.beta_logs) > 0:
        plt.figure(figsize=(6, 5))
        plt.plot(model.beta_logs, label="beta")
        plt.title("Beta over training")
        plt.xlabel("Updates")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/training_beta.png")
        plt.close()
    
    # --- α (temperature) ---
    if model is not None and hasattr(model, "alpha_logs") and len(model.alpha_logs) > 0:
        plt.figure(figsize=(6, 5))
        plt.plot(model.alpha_logs)
        plt.title("SAC Temperature Parameter α")
        plt.xlabel("Updates")
        plt.ylabel("α")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/training_alpha.png")
        plt.close()


    print("[Saved] plotting outputs")

if __name__ == "__main__":
    main()
