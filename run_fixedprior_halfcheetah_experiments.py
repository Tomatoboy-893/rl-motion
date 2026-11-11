"""
先輩版 ADR（FixedPrior）の実験スクリプト（HalfCheetah-v5）
====================================================

位置づけ
----
遅延アンカー対 SAC の正式判定（2026-11-11、成功）が出たあとの
追加ランです。目的は2つあります。

1. 先輩の ADR との直接比較
   先輩の +40%（SAC+ADR 約11,500 vs SAC 約8,200）は論文図の
   読み取り値で、こちらの実験とは物差しが揃っていません。
   先輩版を同一条件・同一コードベースで回すことで、遅延アンカー／
   先輩版 ADR／素の SAC の3手法について、シード別の最終値と
   IQM を直接比べられるようにします。
2. 先輩の結果の再現確認
   こちらの SAC（平均 約9,171、崩壊2本）に対して先輩版が
   どのくらい上乗せするかを見ることで、先輩の +40% が
   再現するかどうかもあわせて確認します。

読み方（先に固定）
------------------------------------------------------------------
新しい「成功／失敗」の基準は作りません。3手法のシード別最終区間
平均・IQM・平均・崩壊数・AUC を並べる記述的な比較に留めます。
SAC との正式判定（固定済み基準による「成功」）はこのランでは
変更しません。

条件（先輩の設定そのまま。手法の中身には一切手を入れない）
------------------------------------------------------------------
- モデル: SACWithFixedPrior（sac_adr_main.py、先輩本人の実装）
- 環境: HalfCheetah-v5 / batch_size=256 / 3,000,000 steps / 5 seeds
- prior_std = 2.0        （事前分布 ρ = N(0, 2.0^2)）
- target_kl = 1.0 / 次元 （クラス内部で行動次元 6 を掛け、合計 6.0）
- beta_kl = 0.01（β の初期値）、beta_lr = 1e-3
- β のクランプは先輩の実装のまま（[1e-12, 1e6]、実質下限なし）。
  遅延アンカー版に付けた下限 1e-4 や KL の上限は、こちらには
  付けません。「先輩の手法そのもの」を測るためです。
- 評価: 5,000 steps ごとに 5 エピソード（決定論的方策）
  （評価と保存の形式は run_halfcheetah_experiments.py と同一）

保存されるログ (npz)
--------------------
./npz_logs_fixed_prior_halfcheetah/fixed_prior_halfcheetah_seed{seed}.npz
共通 : timesteps, returns, pi_entropies, alpha_logs
先輩版: kl_values（KL合計）, beta_logs（βそのものの値。log ではありません）
  ※ 遅延アンカー版にある kl_raw_values や KL の内訳（平均項／σ項）は、
    先輩のクラスが記録していないため、こちらには含まれません。
全シード終了後、集計 npz（fixed_prior_5seed_mean_std.npz、
timesteps / all_returns / mean / std）も同じフォルダに保存します
（先輩の sac_rho_5seed_mean_std.npz と同じ形式）。

所要時間の見通し（概算）
------------------------------------------------------------------
SAC アームの実測が 1シード約7時間、遅延アンカーが約8時間45分
だったため、5シードで約35〜44時間（約2日）の見込みです。
seed 0 の完了時点でペースと kl_values・beta_logs の実測を
確認してください（KL が目標 6.0 付近にいるか、β が消えていないか）。
"""

import os

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_adr_main import SACWithFixedPrior


class FixedUnifiedReturnCallback(EvalCallback):
    """評価周期（eval_freq）のタイミングでのみ重複なくデータを保存する修正コールバック
    （run_halfcheetah_experiments.py と同じもの）"""

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


def run_experiment(seed, total_timesteps=3_000_000, batch_size=256):
    algo_type = "fixed_prior"
    save_dir = f"./npz_logs_{algo_type}_halfcheetah"
    os.makedirs(save_dir, exist_ok=True)

    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    callback = FixedUnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    model = SACWithFixedPrior(
        "MlpPolicy",
        train_env,
        batch_size=batch_size,
        beta_kl=0.01,    # β の初期値（先輩の設定）
        beta_lr=1e-3,    # β の自動調整の学習率（先輩の設定）
        target_kl=1.0,   # KL の目標値（1次元あたり。内部で×行動次元）
        prior_std=2.0,   # 事前分布の σ（先輩の設定）
        seed=seed,
        verbose=0,
    )

    print(f"[FIXED_PRIOR] Starting Seed {seed} (batch_size={batch_size})...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    train_env.close()
    eval_env.close()

    t = np.array(callback.timesteps)
    r = np.array(callback.episode_returns)

    file_path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}.npz"

    save_dict = {
        "timesteps": t,
        "returns": r,
        "pi_entropies": np.array(model.pi_entropies),
        "alpha_logs": np.array(model.alpha_logs),
        "kl_values": np.array(model.kl_values),
        "beta_logs": np.array(model.beta_logs),
    }

    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")


def save_summary_npz(seeds=range(5)):
    """全シードの returns を束ねた集計 npz を保存する
    （先輩の sac_rho_5seed_mean_std.npz と同じ形式）。"""
    save_dir = "./npz_logs_fixed_prior_halfcheetah"
    all_returns, timesteps = [], None
    for seed in seeds:
        path = f"{save_dir}/fixed_prior_halfcheetah_seed{seed}.npz"
        if not os.path.exists(path):
            print(f" [集計] {path} が無いため集計 npz は作りません")
            return
        d = np.load(path)
        if timesteps is None:
            timesteps = np.asarray(d["timesteps"])
        all_returns.append(np.asarray(d["returns"]))
    arr = np.array(all_returns)
    out = f"{save_dir}/fixed_prior_5seed_mean_std.npz"
    np.savez(out, timesteps=timesteps, all_returns=arr,
             mean=arr.mean(axis=0), std=arr.std(axis=0))
    print(f" Saved summary to {out}")


if __name__ == "__main__":
    # ============================================================
    # 本実験用設定（★現在はこちらが有効）
    # ------------------------------------------------------------
    # 先輩版 ADR（FixedPrior）を 5 シード、3,000,000 steps 回します。
    # seed 0 の完了時点で所要時間の見通しが立つので、
    # そこで一度ログと npz を確認してください。
    # ============================================================
    TOTAL_STEPS = 3_000_000

    for seed in range(5):
        print(f"\n==========================================")
        print(f" Running: fixed_prior | Seed: {seed} | BS: 256 | HalfCheetah-v5")
        print(f"==========================================")
        run_experiment(seed, total_timesteps=TOTAL_STEPS, batch_size=256)

    save_summary_npz()

    # ============================================================
    # 試し回し用設定（確認用に残してあります。通常は使いません）
    # ============================================================
    # TOTAL_STEPS = 100_000
    # print(f"\n==========================================")
    # print(f" Running (TRIAL): fixed_prior | Seed: 0 | BS: 256 | HalfCheetah-v5")
    # print(f"==========================================")
    # run_experiment(0, total_timesteps=TOTAL_STEPS, batch_size=256)
