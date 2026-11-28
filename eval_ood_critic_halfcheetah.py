#
# eval_ood_critic_halfcheetah.py
#
# 機構分析・項目3「訪問した分布の外でのクリティックの値」の測定スクリプト。
#
# 目的:
#   ステップ2で「崩壊したクリティックは、バッファ内（這い歩行で実際に訪れた
#   範囲）では自己整合している」ことまでは分かった。本スクリプトは、その範囲の
#   外、つまり健全な方策が訪れる状態と行動に対して、崩壊側のクリティックが
#   どんな Q を返すかを測る。学習は一切行わない（評価のみ）。
#
# 読み分け（実行前に固定）:
#   崩壊側クリティックが、健全側の状態で「健全な行動」の Q を「這い行動」の Q
#   より高く返すなら、改善の信号はクリティックの中に存在するのに方策が使えて
#   いない（信号はあるが未使用）。
#   逆に、這い行動と同等かそれ以上にしか返さないなら、クリティック自体が
#   逃げる方向を低く見積もっており、改善の勾配が存在しない（価値の汎化の失敗
#   が罠を支えている）。
#   対照として、健全側クリティックでも同じ表を作る（健全側は健全な行動を
#   高く返すはずで、測定系の健全性チェックになる）。
#
# 標本:
#   ステップ2（run_critic_probe_halfcheetah.py）のチェックポイントを使う。
#   ./npz_logs_critic_probe_halfcheetah/checkpoints/
#     {pure_sac|fixed_prior}_halfcheetah_seed{7|8}_criticprobe_{500k|1000k}.zip
#   seed 8 = 這い型（両手法）、seed 7 = 健全（両手法）。
#
# 状態の集め方:
#   各方策を新しい環境（評価用シード）で確率的に走らせ、訪れた状態と実際に
#   取った行動を記録する。その状態に対して、もう一方の方策の行動（決定論的
#   ＝平均行動と、確率的サンプル）も計算し、両方のクリティックで Q を測る。
#
# 出力:
#   ./plots_ood_critic/ood_critic_summary.csv とコンソールへの要約表示。
#   バージョン情報（torch / SB3 / Gymnasium）も要約に含める。
#
# 使い方:
#   python3 eval_ood_critic_halfcheetah.py                 # 1000k で両手法
#   python3 eval_ood_critic_halfcheetah.py --steps 500k    # 500k 時点
#   python3 eval_ood_critic_halfcheetah.py --algo pure_sac # SAC のみ
#
import argparse
import csv
import os
import sys

import numpy as np

import gymnasium as gym
import torch
from stable_baselines3.common.env_util import make_vec_env

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior

DEFAULT_CKPT_DIR = "./npz_logs_critic_probe_halfcheetah/checkpoints"
DEFAULT_OUT_DIR = "./plots_ood_critic"

SPECIMENS = [("pure_sac", 8), ("pure_sac", 7),
             ("fixed_prior", 8), ("fixed_prior", 7)]
ROLE = {8: "collapsed", 7: "healthy"}


class _LoadableFixedPrior(SACWithFixedPrior):
    """先輩版は必須引数にデフォルトが無く SB3 の load が再構築に失敗するため、
    ステップ3と同じ読み込み専用サブクラスを使う（先輩版クラス自体は無変更）。"""

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


def collect_states(model, n_states, env_seed):
    """方策を確率的に走らせ、(状態, 実際に取った行動) を集める。"""
    env = gym.make("HalfCheetah-v5")
    obs_list, act_list = [], []
    obs, _ = env.reset(seed=env_seed)
    while len(obs_list) < n_states:
        action, _ = model.predict(obs, deterministic=False)
        obs_list.append(np.asarray(obs, dtype=np.float32))
        act_list.append(np.asarray(action, dtype=np.float32))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()
    return np.array(obs_list), np.array(act_list)


def batch_actions(model, states, deterministic):
    """状態の配列に対して方策の行動をまとめて計算する。"""
    acts = []
    for s in states:
        a, _ = model.predict(s, deterministic=deterministic)
        acts.append(np.asarray(a, dtype=np.float32))
    return np.array(acts)


def q_values(model, states, actions):
    """クリティックの Q1 / Q2 を返す（観測・行動は numpy 配列）。"""
    critic = model.critic
    device = model.device
    with torch.no_grad():
        obs_t = torch.as_tensor(states, dtype=torch.float32, device=device)
        act_t = torch.as_tensor(actions, dtype=torch.float32, device=device)
        try:
            # SB3 の ContinuousCritic は forward(obs, action) で Q の組を返す
            out = critic(obs_t, act_t)
            q1, q2 = out[0], out[1]
        except TypeError:
            # フォールバック: q_networks は連結した入力を受ける Sequential
            joined = torch.cat([obs_t, act_t], dim=1)
            q1 = critic.q_networks[0](joined)
            q2 = critic.q_networks[1](joined)
    return (q1.cpu().numpy().reshape(-1), q2.cpu().numpy().reshape(-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    parser.add_argument("--steps", default="1000k",
                        choices=["500k", "1000k"])
    parser.add_argument("--algo", default="both",
                        choices=["both", "pure_sac", "fixed_prior"])
    parser.add_argument("--n-states", type=int, default=4096)
    parser.add_argument("--env-seed", type=int, default=777,
                        help="状態収集用 env のシード（標本のシードと無関係な中立値）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    import stable_baselines3
    print(f"[versions] torch={torch.__version__} "
          f"sb3={stable_baselines3.__version__} gym={gym.__version__}")

    algos = ["pure_sac", "fixed_prior"] if args.algo == "both" else [args.algo]
    dummy_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=0)

    rows = []
    for algo in algos:
        models = {}
        states = {}
        own_actions = {}
        for seed in (8, 7):
            models[seed] = load_model(algo, seed, args.steps,
                                      args.ckpt_dir, dummy_env)
            s, a = collect_states(models[seed], args.n_states, args.env_seed)
            states[seed] = s
            own_actions[seed] = a
            print(f"[{algo}] seed {seed} ({ROLE[seed]}): 状態 {len(s)} 件を収集")

        # 状態の出所 × 行動の出所 × クリティック の全組み合わせを測る
        for critic_seed in (8, 7):
            critic_model = models[critic_seed]
            for state_seed in (8, 7):
                s = states[state_seed]
                action_sets = {
                    "own_taken": own_actions[state_seed],
                    "other_det": batch_actions(models[7 if state_seed == 8 else 8],
                                               s, deterministic=True),
                    "other_sample": batch_actions(models[7 if state_seed == 8 else 8],
                                                  s, deterministic=False),
                    "self_det": batch_actions(models[state_seed], s,
                                              deterministic=True),
                }
                for act_name, acts in action_sets.items():
                    q1, q2 = q_values(critic_model, s, acts)
                    qmin = np.minimum(q1, q2)
                    rows.append({
                        "algo": algo,
                        "critic": f"seed{critic_seed}_{ROLE[critic_seed]}",
                        "state_source": f"seed{state_seed}_{ROLE[state_seed]}",
                        "action_source": act_name,
                        "q1_mean": float(q1.mean()),
                        "q2_mean": float(q2.mean()),
                        "qmin_mean": float(qmin.mean()),
                        "qmin_std": float(qmin.std()),
                        "n": int(len(s)),
                    })

        # 核心の対比を表示する: 健全側の状態で、健全な行動 vs 這い行動
        def get(critic_seed, state_seed, act_name):
            for r in rows:
                if (r["algo"] == algo
                        and r["critic"] == f"seed{critic_seed}_{ROLE[critic_seed]}"
                        and r["state_source"] == f"seed{state_seed}_{ROLE[state_seed]}"
                        and r["action_source"] == act_name):
                    return r["qmin_mean"]
            return float("nan")

        print(f"\n===== [{algo}] 核心の対比（Qmin 平均, steps={args.steps}） =====")
        for critic_seed in (8, 7):
            q_healthy_act = get(critic_seed, 7, "own_taken")
            q_crawl_act = get(critic_seed, 7, "other_det")
            print(f" クリティック seed {critic_seed}（{ROLE[critic_seed]}）"
                  f"／健全側の状態: 健全な行動の Q = {q_healthy_act:.1f}, "
                  f"這い行動（決定論）の Q = {q_crawl_act:.1f}, "
                  f"差 = {q_healthy_act - q_crawl_act:+.1f}")
            q_crawl_own = get(critic_seed, 8, "own_taken")
            q_healthy_on_crawl = get(critic_seed, 8, "other_det")
            print(f" クリティック seed {critic_seed}（{ROLE[critic_seed]}）"
                  f"／這い側の状態: 這い行動の Q = {q_crawl_own:.1f}, "
                  f"健全な行動（決定論）の Q = {q_healthy_on_crawl:.1f}, "
                  f"差 = {q_healthy_on_crawl - q_crawl_own:+.1f}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "ood_critic_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[out] {out_path} に {len(rows)} 行を保存しました")
    print("読み分け: 崩壊側クリティックで「健全な行動の Q − 這い行動の Q」が"
          "明確にプラスなら信号はあるが未使用、マイナス〜ゼロなら価値の汎化の"
          "失敗が罠を支えている、が事前に固定した解釈です。")


if __name__ == "__main__":
    main()
