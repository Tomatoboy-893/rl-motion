"""
SAC／先輩版 ADR の 10 シード拡張ラン（HalfCheetah-v5、seed 5〜9）
====================================================

位置づけ
----
2026-11-18 の教授相談での方向付けを受けた実験です。

- 研究のゴールは性能の数字ではなくメカニズムの理解で、特に
  「失敗シード（上手くいかなかったシード）」の解明が主題。
- 教授の問い：失敗パターンは何種類あるか／シードを増やすと
  失敗の傾向は現れるか／最初にニューラルネットが吐き出す
  初期分布はどうなっているか。
- 今後の実験は SAC と ADR（先輩版）のみに限り、条件は
  青柳先輩の実行条件に合わせる（ユーザー決定、2026-11-18）。

このスクリプトは、既存の本実験（seed 0〜4）に seed 5〜9 を追加し、
両手法を 10 シードに揃えるものです。失敗の発生率と失敗標本を
増やすことが目的で、新しい成功／失敗の基準は作りません
（記述的な分析に留めます）。

条件（既存ランと完全に同一。手法の中身には一切手を入れない）
------------------------------------------------------------------
- 環境: HalfCheetah-v5 / batch_size=256 / 3,000,000 steps
  （batch_size=256 は先輩の実行条件＝SB3 既定への推定に基づく。
    seed 0〜4 と同じ値なので、10 シードを同一条件として束ねられる）
- pure_sac   : PureSACWithLogging（sac_models.py、素の SAC）
- fixed_prior: SACWithFixedPrior（sac_adr_main.py、先輩本人の実装）
  prior_std=2.0 / target_kl=1.0（次元あたり）/ beta_kl=0.01 / beta_lr=1e-3
  β のクランプ等も先輩の実装のまま
- 評価: 5,000 steps ごとに 5 エピソード（決定論的方策）
- 保存先・ファイル名・npz のキーは既存スクリプトと同一
  （./npz_logs_pure_sac_halfcheetah / ./npz_logs_fixed_prior_halfcheetah）

実行順序
------------------------------------------------------------------
seed を外側のループにし、1 シードごとに SAC → 先輩版 の順で回します。
途中で止めても両手法のシード数が揃った状態で残るための配慮です。

初期分布の記録（このスクリプトで追加した唯一の計測）
------------------------------------------------------------------
各ランの学習開始直前に、初期化されたばかりの方策が「どんな分布を
吐くか」を記録します（教授の問いへの対応）。

- 別の探査用環境でランダム行動を 2,000 steps 回して状態を集める
- その状態に対して、方策ネットワークの分布パラメータ
  （tanh 前のガウス分布の平均 μ と標準偏差 σ）を実測する
- 生データ（mu_samples / sigma_samples、2000×6）と要約統計を
  ./npz_logs_initdist_halfcheetah/{algo}_halfcheetah_seed{seed}_initdist.npz
  に保存する
- 学習への影響はありません：探査は学習用とは別の環境で行い、
  方策側もパラメータの読み取りのみでサンプリングはしません。
  記録に失敗しても学習は止めず、警告を出して続行します。

所要時間の見通し（概算）
------------------------------------------------------------------
実測では SAC 約7時間／シード、先輩版 約7〜8.75時間／シード。
10 ラン（2手法×5シード）で約3〜4日の見込みです。
DroQ 本ランのプロセスが残っている場合は、先に停止してから
起動してください（計算機を共有するため）。
"""

import os
import sys

import numpy as np
import gymnasium as gym

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior

# 試し回し（True にすると 10,000 steps・保存先は smoke 用フォルダ）
SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"


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


def snapshot_initial_distribution(model, algo_type, seed, n_steps=2000):
    """学習前の方策が吐く分布（tanh 前ガウス分布の μ・σ）を実測して保存する。

    学習には一切影響させない。失敗しても例外を外に出さず警告のみ。
    """
    try:
        import torch

        probe_env = gym.make("HalfCheetah-v5")
        obs, _ = probe_env.reset(seed=seed * 1000 + 777)
        states = []
        for _ in range(n_steps):
            states.append(obs)
            obs, _, terminated, truncated, _ = probe_env.step(
                probe_env.action_space.sample()
            )
            if terminated or truncated:
                obs, _ = probe_env.reset()
        probe_env.close()
        states = np.array(states, dtype=np.float32)

        actor = model.policy.actor
        mus, sigmas = [], []
        with torch.no_grad():
            for i in range(0, len(states), 512):
                batch = states[i:i + 512]
                obs_tensor, _ = model.policy.obs_to_tensor(batch)
                mean_actions, log_std, _ = actor.get_action_dist_params(obs_tensor)
                mus.append(mean_actions.cpu().numpy())
                sigmas.append(np.exp(log_std.cpu().numpy()))
        mu = np.concatenate(mus)
        sigma = np.concatenate(sigmas)

        save_dir = "./npz_logs_initdist_halfcheetah"
        if SMOKE:
            save_dir = "./npz_logs_smoke_10seed_ext"
        os.makedirs(save_dir, exist_ok=True)
        path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}_initdist.npz"
        np.savez(
            path,
            mu_samples=mu,
            sigma_samples=sigma,
            mu_mean_per_dim=mu.mean(axis=0),
            mu_std_per_dim=mu.std(axis=0),
            sigma_mean_per_dim=sigma.mean(axis=0),
            sigma_std_per_dim=sigma.std(axis=0),
            sigma_min=float(sigma.min()),
            sigma_max=float(sigma.max()),
        )
        print(f" [initdist] saved {path} "
              f"(sigma mean per dim: {np.round(sigma.mean(axis=0), 4)})")
    except Exception as e:  # 記録の失敗で本ランを止めない
        print(f" [initdist] WARNING: 初期分布の記録に失敗しました（続行します）: {e}")


def run_experiment(algo_type, seed, total_timesteps=3_000_000, batch_size=256):
    save_dir = f"./npz_logs_{algo_type}_halfcheetah"
    if SMOKE:
        save_dir = "./npz_logs_smoke_10seed_ext"
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

    if algo_type == "pure_sac":
        model = PureSACWithLogging(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            seed=seed,
            verbose=0,
        )
    elif algo_type == "fixed_prior":
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
    else:
        raise ValueError(f"unknown algo_type: {algo_type}")

    # 学習開始直前に初期分布を記録（学習には影響しない）
    snapshot_initial_distribution(model, algo_type, seed)

    print(f"[{algo_type.upper()}] Starting Seed {seed} (batch_size={batch_size})...")
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
    }
    if algo_type == "fixed_prior":
        save_dict.update({
            "kl_values": np.array(model.kl_values),
            "beta_logs": np.array(model.beta_logs),
        })

    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")


def save_10seed_summaries():
    """seed 0〜9 が揃った手法について、10 シード集計 npz を保存する。"""
    for algo_type in ("pure_sac", "fixed_prior"):
        save_dir = f"./npz_logs_{algo_type}_halfcheetah"
        all_returns, timesteps = [], None
        ok = True
        for seed in range(10):
            path = f"{save_dir}/{algo_type}_halfcheetah_seed{seed}.npz"
            if not os.path.exists(path):
                print(f" [集計] {path} が無いため {algo_type} の集計は作りません")
                ok = False
                break
            d = np.load(path)
            if timesteps is None:
                timesteps = np.asarray(d["timesteps"])
            all_returns.append(np.asarray(d["returns"]))
        if ok:
            arr = np.array(all_returns)
            out = f"{save_dir}/{algo_type}_10seed_mean_std.npz"
            np.savez(out, timesteps=timesteps, all_returns=arr,
                     mean=arr.mean(axis=0), std=arr.std(axis=0))
            print(f" Saved summary to {out}")


if __name__ == "__main__":
    if SMOKE:
        # 試し回し: seed 5 の SAC を 10,000 steps だけ（保存先は smoke 用）
        print("=== SMOKE MODE (10,000 steps, seed 5, pure_sac only) ===")
        run_experiment("pure_sac", 5, total_timesteps=10_000, batch_size=256)
    else:
        TOTAL_STEPS = 3_000_000
        for seed in range(5, 10):
            for algo_type in ("pure_sac", "fixed_prior"):
                print(f"\n==========================================")
                print(f" Running: {algo_type} | Seed: {seed} | BS: 256 | HalfCheetah-v5")
                print(f"==========================================")
                run_experiment(algo_type, seed, total_timesteps=TOTAL_STEPS, batch_size=256)
        save_10seed_summaries()
