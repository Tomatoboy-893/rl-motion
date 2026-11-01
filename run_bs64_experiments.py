import os
import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_models import PureSACWithLogging, SACWithPolicyMeanPrior


class FixedUnifiedReturnCallback(EvalCallback):
    """評価周期（eval_freq）のタイミングでのみ重複なくデータを保存する修正コールバック"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_returns = []
        self.timesteps = []

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.episode_returns.append(self.last_mean_reward)
                self.timesteps.append(self.num_timesteps)
        return result


def run_experiment(algo_type, seed, total_timesteps=3_000_000, batch_size=64):
    save_dir = f"./npz_logs_{algo_type}_bs64"
    os.makedirs(save_dir, exist_ok=True)

    train_env = make_vec_env("Humanoid-v5", n_envs=1, seed=seed)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=seed + 1000)

    callback = FixedUnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    if algo_type == "policy_mean_prior":
        model = SACWithPolicyMeanPrior(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            beta_kl=0.01,
            beta_lr=1e-3,
            target_kl=1.0,
            prior_std=0.3,
            seed=seed,
            verbose=0
        )
    elif algo_type == "pure_sac":
        model = PureSACWithLogging(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            seed=seed,
            verbose=0
        )

    print(f"[{algo_type.upper()}] Starting Seed {seed} (batch_size={batch_size})...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    file_path = f"{save_dir}/{algo_type}_bs64_seed{seed}.npz"

    # 両アルゴリズム共通で保存する項目
    save_dict = {
        "timesteps": t,
        "returns": r,
        "pi_entropies": np.array(model.pi_entropies),
        "alpha_logs": np.array(model.alpha_logs),
    }

    # 提案手法固有の保存項目
    if algo_type == "policy_mean_prior":
        save_dict.update({
            "kl_values": np.array(model.kl_values),
            "beta_logs": np.array(model.beta_logs),
        })

    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")


if __name__ == "__main__":
    TOTAL_STEPS = 3_000_000

    # 提案手法と Vanilla SAC の全5シード（batch_size=64）を一括実行
    for algo in ["policy_mean_prior", "pure_sac"]:
        for seed in range(5):
            print(f"\n==========================================")
            print(f" Running: {algo} | Seed: {seed} | BS: 64")
            print(f"==========================================")
            run_experiment(algo, seed, total_timesteps=TOTAL_STEPS, batch_size=64)
