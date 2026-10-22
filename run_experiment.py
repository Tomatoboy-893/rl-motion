import os
import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

# 1つ目のファイルからカスタムクラスをインポート
from sac_policy_mean_prior import SACWithPolicyMeanPrior

# 保存先のディレクトリ設定
SAVE_DIR = "./npz_logs_policy_mean_prior"
os.makedirs(SAVE_DIR, exist_ok=True)


class UnifiedReturnCallback(EvalCallback):
    """評価ごとの平均報酬とタイムステップを記録するコールバック"""
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


def run_experiment(seed, total_timesteps):
    # 学習用・評価用環境の作成
    train_env = make_vec_env("Humanoid-v5", n_envs=1, seed=seed)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=seed + 1000)

    callback = UnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # カスタムモデルの初期化
    model = SACWithPolicyMeanPrior(
        "MlpPolicy", 
        train_env,
        beta_kl=0.01, 
        beta_lr=1e-3, 
        target_kl=1.0,
        prior_std=1.0, 
        seed=seed,
        verbose=0
    )

    # 学習の実行
    print(f"Starting training for Seed {seed}...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    train_env.close()
    eval_env.close()

    # 評価結果と内部指標をNumpy配列に変換
    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    # ★ 訓練完了後、すべてのログデータを一括で .npz ファイルとして保存
    file_path = f"{SAVE_DIR}/policy_mean_prior_seed{seed}.npz"
    np.savez(
        file_path,
        timesteps=t,
        returns=r,
        kl_values=np.array(model.kl_values),
        pi_entropies=np.array(model.pi_entropies),
        beta_logs=np.array(model.beta_logs),
        alpha_logs=np.array(model.alpha_logs),
        prior_mean_logs=np.array(model.prior_mean_logs)
    )
    print(f"[Saved] Metrics and returns successfully saved to {file_path}")

    return t, r, model


def main():
    TOTAL_STEPS = 3_000_000
    
    # 5シード（seed 0 〜 4）で実験を回す
    for seed in range(5):
        print(f"\n==============================")
        print(f"=== Policy-Mean Prior: Seed {seed} ===")
        print(f"==============================")
        run_experiment(seed, TOTAL_STEPS)


if __name__ == "__main__":
    main()
