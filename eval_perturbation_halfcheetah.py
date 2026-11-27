#
# eval_perturbation_halfcheetah.py
#
# 機構分析・項目6「這い歩行が真の局所最適かどうか」の測定スクリプト。
#
# 目的:
#   這い方策（崩壊標本）を実環境で走らせ、その行動を少しだけ崩したときに
#   リターンがどう動くかを測る。学習は一切行わない（評価のみ）。
#   ステップ2・3で分かったのは「学習の力学にとっての固定点」という振る舞い
#   までで、真のリターンの地形で這い歩行の近傍に改善方向があるかは未測定
#   だった。本スクリプトはその測定をする。
#
# 摂動の種類:
#   (a) 行動ノイズ: 這い方策の決定論的行動にガウスノイズ（σ = 0.1 / 0.3 /
#       0.5 / 1.0）を足す。小さいノイズで大きく改善するなら、近傍に改善方向
#       がある。
#   (b) 方策の混合: 各ステップで這い方策と健全方策の決定論的行動を
#       a = (1-λ) * a_這い + λ * a_健全 で混ぜる（λ = 0.25 / 0.5 / 0.75 /
#       1.0）。λ が小さいうちに跳ねるなら盆地は狭く、λ→1 でしか抜けない
#       なら盆地は方策空間で広い。
#   対照: 這い方策そのまま（λ=0）と健全方策そのまま。
#
# 読み分け（実行前に固定）:
#   小さい摂動で大きく改善 → 真の地形でも局所最適ではない。罠は学習力学と
#     価値評価の側にある（項目3の結果と合わせて読む）。
#   どの摂動でも這いの帯域（〜2,600）から出ず、健全方策そのものでしか
#     抜けない → 這い歩行の盆地は広く、真の局所最適という読みと整合する。
#   留保: 測定はすべてエピソード開始からの走行。這い方策が訪れた途中の
#     状態から始めた場合の挙動は、この測定では分からない。
#
# 標本:
#   ステップ2のチェックポイント（項目3と同じファイル）:
#   ./npz_logs_critic_probe_halfcheetah/checkpoints/
#     {pure_sac|fixed_prior}_halfcheetah_seed{7|8}_criticprobe_{500k|1000k}.zip
#
# 出力:
#   ./plots_perturbation/perturbation_summary.csv とコンソールへの要約。
#
# 使い方:
#   python3 eval_perturbation_halfcheetah.py                  # 1000k, 両手法
#   python3 eval_perturbation_halfcheetah.py --steps 500k --episodes 20
#
import argparse
import csv
import os

import numpy as np

import gymnasium as gym
import torch
from stable_baselines3.common.env_util import make_vec_env

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior

DEFAULT_CKPT_DIR = "./npz_logs_critic_probe_halfcheetah/checkpoints"
DEFAULT_OUT_DIR = "./plots_perturbation"

NOISE_SIGMAS = [0.1, 0.3, 0.5, 1.0]
MIX_LAMBDAS = [0.25, 0.5, 0.75, 1.0]


class _LoadableFixedPrior(SACWithFixedPrior):
    """ステップ3と同じ読み込み専用サブクラス（先輩版クラス自体は無変更）。"""

    def __init__(self, *args, beta_kl=0.01, beta_lr=1e-3,
                 target_kl=1.0, prior_std=2.0, **kwargs):
        super().__init__(*args, beta_kl=beta_kl, beta_lr=beta_lr,
                         target_kl=target_kl, prior_std=prior_std, **kwargs)


def load_model(algo_type, seed, steps_label, ckpt_dir, env):
    path = os.path.join(
        ckpt_dir, f"{algo_type}_halfcheetah_seed{seed}_criticprobe_{steps_label}.zip")
    if not os.path.exists(path):
        raise FileNotFoundError(f"チェックポイントがありません: {path}")
    cls = _LoadableFixedPrior if algo_type == "fixed_prior" else PureSACWithLogging
    model = cls.load(path, env=env)
    model.set_env(env)
    return model


def det_action(model, obs):
    a, _ = model.predict(obs, deterministic=True)
    return np.asarray(a, dtype=np.float64)


def run_episode(crawl_model, healthy_model, condition, param, ep_seed, rng):
    """1エピソード走らせてリターンを返す。"""
    env = gym.make("HalfCheetah-v5")
    obs, _ = env.reset(seed=ep_seed)
    total = 0.0
    done = False
    while not done:
        a_crawl = det_action(crawl_model, obs)
        if condition == "base_crawl":
            action = a_crawl
        elif condition == "noise":
            action = a_crawl + rng.normal(0.0, param, size=a_crawl.shape)
        elif condition == "mix":
            a_healthy = det_action(healthy_model, obs)
            action = (1.0 - param) * a_crawl + param * a_healthy
        elif condition == "base_healthy":
            action = det_action(healthy_model, obs)
        else:
            raise ValueError(condition)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    env.close()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--steps", default="1000k",
                        choices=["500k", "1000k"])
    parser.add_argument("--algo", default="both",
                        choices=["both", "pure_sac", "fixed_prior"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--env-seed", type=int, default=777,
                        help="評価エピソードのシード基点（条件間で共通にして対照にする）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    import stable_baselines3
    print(f"[versions] torch={torch.__version__} "
          f"sb3={stable_baselines3.__version__} gym={gym.__version__}")

    algos = ["pure_sac", "fixed_prior"] if args.algo == "both" else [args.algo]
    dummy_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=0)

    conditions = ([("base_crawl", 0.0)]
                  + [("noise", s) for s in NOISE_SIGMAS]
                  + [("mix", l) for l in MIX_LAMBDAS]
                  + [("base_healthy", 0.0)])

    rows = []
    for algo in algos:
        crawl = load_model(algo, 8, args.steps, args.ckpt_dir, dummy_env)
        healthy = load_model(algo, 7, args.steps, args.ckpt_dir, dummy_env)
        for condition, param in conditions:
            returns = []
            for ep in range(args.episodes):
                rng = np.random.default_rng(args.env_seed * 1000 + ep)
                r = run_episode(crawl, healthy, condition, param,
                                args.env_seed + ep, rng)
                returns.append(r)
            mean_r = float(np.mean(returns))
            std_r = float(np.std(returns))
            rows.append({
                "algo": algo, "condition": condition, "param": param,
                "mean_return": mean_r, "std_return": std_r,
                "n_episodes": args.episodes,
            })
            print(f"[{algo}] {condition:12s} param={param:<5} "
                  f"return = {mean_r:9.1f} +/- {std_r:7.1f} "
                  f"(n={args.episodes})")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "perturbation_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[out] {out_path} に {len(rows)} 行を保存しました")
    print("読み分け: 小さい摂動で大きく改善するなら真の局所最適ではない。"
          "健全方策そのものでしか抜けないなら盆地は広い、が事前に固定した"
          "解釈です（エピソード開始からの測定という留保つき）。")


if __name__ == "__main__":
    main()
