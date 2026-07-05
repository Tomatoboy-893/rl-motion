# sac_entropy_run.py
import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from sac_entropy import SACWithEntropyLogging

SAVE_DIR = "./npz_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# callback
# =========================
class Callback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rets = []
        self.ts = []

    def _on_step(self):
        super()._on_step()
        if self.last_mean_reward is not None:
            self.rets.append(self.last_mean_reward)
            self.ts.append(self.num_timesteps)
        return True


# =========================
# env
# =========================
def make_env(seed):
    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)
    return train_env, eval_env


# =========================
# run single
# =========================
def run(seed, steps):

    train_env, eval_env = make_env(seed)

    cb = Callback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # ★青柳先輩仕様：seedはモデルに渡さない
    model = SACWithEntropyLogging(
        "MlpPolicy",
        train_env,
        verbose=0,
    )

    model.learn(steps, callback=cb)

    np.savez(
        f"{SAVE_DIR}/sac_seed{seed}.npz",
        returns=np.array(cb.rets),
        timesteps=np.array(cb.ts),
    )

    np.savez(
        f"{SAVE_DIR}/sac_entropy_seed{seed}.npz",
        entropy=np.array(model.pi_entropies),
        alpha=np.array(model.alpha_logs),
    )

    train_env.close()
    eval_env.close()

    return cb.ts, cb.rets, model.pi_entropies


# =========================
# sweep
# =========================
def main():

    STEPS = 1_000_000
    seeds = range(5)

    all_returns = []
    all_entropy = []

    for s in seeds:
        print("seed:", s)
        t, r, e = run(s, STEPS)
        all_returns.append(r)
        all_entropy.append(e)

    # =========================
    # align
    # =========================
    min_len = min(len(x) for x in all_returns)
    all_returns = np.array([x[:min_len] for x in all_returns])

    mean = all_returns.mean(axis=0)
    std = all_returns.std(axis=0)

    t = np.array(t[:min_len])

    # =========================
    # save stats
    # =========================
    np.savez(
        f"{SAVE_DIR}/sac_5seed_mean_std.npz",
        timesteps=t,
        mean=mean,
        std=std,
        all_returns=all_returns,
    )

    # =========================
    # plot return
    # =========================
    plt.figure()
    plt.plot(t, mean)
    plt.fill_between(t, mean-std, mean+std, alpha=0.3)
    plt.title("SAC Return")
    plt.savefig(f"{SAVE_DIR}/sac_return.png")
    plt.close()

    # =========================
    # entropy plot
    # =========================
    plt.figure()
    for e in all_entropy:
        plt.plot(e, alpha=0.3)
    plt.title("Policy Entropy")
    plt.savefig(f"{SAVE_DIR}/sac_entropy.png")
    plt.close()

    print("DONE")


if __name__ == "__main__":
    main()
