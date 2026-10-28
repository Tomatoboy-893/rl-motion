import os
import time
import numpy as np
import torch
from torch.distributions import Normal
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

SAVE_DIR = "./npz_logs_pure_sac_bs256"
os.makedirs(SAVE_DIR, exist_ok=True)


class SACMetricsCallback(EvalCallback):
    """評価タイミング(5,000ステップごと)に returns, pi_entropies, kl_values を記録"""
    def __init__(self, model_ref=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_ref = model_ref
        self.timesteps = []
        self.returns = []
        self.pi_entropies = []
        self.kl_values = []

    @staticmethod
    def kl_to_standard_normal(mu, log_std):
        var = torch.exp(2 * log_std)
        kl = -log_std + 0.5 * (var + mu ** 2 - 1.0)
        return kl.sum(dim=-1).mean().item()

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.last_mean_reward is not None:
                self.timesteps.append(self.num_timesteps)
                self.returns.append(self.last_mean_reward)

                if self.model_ref is not None and hasattr(self.model_ref, "replay_buffer"):
                    if self.model_ref.replay_buffer.size() > 0:
                        with torch.no_grad():
                            try:
                                replay_data = self.model_ref.replay_buffer.sample(256)
                                obs = replay_data.observations
                                pi_mean, pi_log_std, _ = self.model_ref.actor.get_action_dist_params(obs)
                                pi_log_std = torch.clamp(pi_log_std, -20.0, 2.0)
                                pi_dist = Normal(pi_mean, torch.exp(pi_log_std))

                                # 1. エントロピー
                                entropy = pi_dist.entropy().sum(dim=-1).mean().item()
                                self.pi_entropies.append(entropy)

                                # 2. KLダイバージェンス (対 N(0, I))
                                kl = self.kl_to_standard_normal(pi_mean, pi_log_std)
                                self.kl_values.append(kl)
                            except Exception:
                                pass
        return result


def make_envs(seed):
    train_env = make_vec_env("Humanoid-v5", n_envs=1, seed=seed)
    eval_env = gym.make("Humanoid-v5")
    eval_env.reset(seed=seed + 1000)
    return train_env, eval_env


def main():
    TOTAL_STEPS = 3_000_000
    NUM_SEEDS = 5

    start_time = time.time()
    print("=========================================")
    print(" Humanoid-v5 Vanilla SAC Sweep")
    print(" Conditions: n_envs=1, batch_size=256, eval_freq=5000")
    print("=========================================")

    for seed in range(NUM_SEEDS):
        print(f"\n--- [Vanilla SAC] Seed {seed} ---")
        train_env, eval_env = make_envs(seed)

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            batch_size=256,       # Policy-Mean 側と完全一致
            seed=seed,            # シード固定
            verbose=0,
            device="cuda"
        )

        callback = SACMetricsCallback(
            model_ref=model,
            eval_env=eval_env,
            eval_freq=5000,      # 5,000ステップごと
            n_eval_episodes=5,
            deterministic=True,
        )

        model.learn(total_timesteps=TOTAL_STEPS, callback=callback)

        prefix = f"pure_sac_bs256_seed{seed}"
        file_path = f"{SAVE_DIR}/{prefix}.npz"

        # 指定指標のみ保存
        np.savez(
            file_path,
            timesteps=np.array(callback.timesteps),
            returns=np.array(callback.returns),
            pi_entropies=np.array(callback.pi_entropies),
            kl_values=np.array(callback.kl_values)
        )
        model.save(f"{SAVE_DIR}/{prefix}_model.zip")
        print(f"💾 [Saved] {file_path}")

        train_env.close()
        eval_env.close()

    end_time = time.time()
    print(f"\n🎉 完了！ 総所要時間: {(end_time - start_time) / 3600:.2f} 時間")


if __name__ == "__main__":
    main()
