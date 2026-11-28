#!/usr/bin/env python3
"""
項目4: α を戻す介入（α の低さが崩壊の維持因かどうか）
================================================================

位置づけ（2026-11-28 設計確定）
--------------------------------
未解決6項目の項目4。ステップ0と拡張実験で、α の水準は両手法とも
成否と強く分離した（100–500k の時点で分離が完了し、重なりなし）。
しかしステップ3の介入はすべて α を保持して設計したため、α を
戻したことは一度もない。因果（維持因か、単なる目印か）は未解決。

本スクリプトは、ステップ3と同じ保存状態（500k 時点のモデル＋
バッファ、seed 8 ペア）から分岐し、α だけを操作して 1M まで
継続する。設計メモ: 項目4設計_α介入_2026-11-28.md

この介入で決まるのは「維持因かどうか」まで。序盤の α が引き金
だったかは試せない（項目1の領分）。

条件（標本ごとに2条件）
------------------------
- alpha_reset : log α を目標値に置き直し、α 用オプティマイザの
                蓄積状態を消す。自動調整は継続する。
- alpha_fixed : log α を目標値に置き直し、α 用オプティマイザの
                学習率を 0 にして α を全期間固定する。

目標値（設計前に存在した数値から固定）
--------------------------------------
同手法の健全シード（seed 7）の元ランにおける 100–500k 窓の
α 平均（ステップ0の集計 distribution_per_seed.csv より）:
- pure_sac   : 0.1581
- fixed_prior: 0.1226
分岐点での α は pure_sac 0.04086 / fixed_prior 0.01577。

単一変数の線引き
----------------
- 操作するのは α とその調整装置の状態だけ。アクター・
  クリティック・バッファは保持する。先輩版の β も保持する。
- 対照はステップ3の control（同一の保存状態から分岐済み）を
  流用する。本スクリプトでは control を回さない。

判定基準（実行前に固定）
------------------------
継続の最終区間（900k–1M）の評価リターン平均で判定し、4,000 以上
で救済とする（ステップ3と同じ）。alpha_fixed では記録された α が
全期間目標値のままであることを集計時に検証する。

条件（ステップ3と同一）
------------------------
- 環境: HalfCheetah-v5 / batch_size=256
- 評価: 5,000 steps ごとに 5 エピソード（決定論的）。
  train seed = seed、eval seed = seed+1000。
- プローブ記録: ステップ2・3と同じ CriticProbeCallback。
- 状態の置き場所: ./npz_logs_step3_halfcheetah/state/（読み取りのみ）
- 出力: ./npz_logs_alpha_halfcheetah/
  {algo}_halfcheetah_seed{seed}_{condition}.npz
  （ステップ3と同じキー。timesteps は 500k から 1M の絶対値）

使い方（keisanB の rl-motion 直下で）
-------------------------------------
# 試し回し（10k 状態から 2k 継続、両手法×2条件。ステップ3の
# 試し回しで作った 10k 状態を使う）
python3 run_alpha_intervention_halfcheetah.py --smoke

# 介入1本
python3 run_alpha_intervention_halfcheetah.py --intervene --algo pure_sac --condition alpha_reset

# 全部を順に（4本、約6時間）
python3 run_alpha_intervention_halfcheetah.py --plan-alpha
"""

import argparse
import math
import os

import numpy as np
import torch

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior
from run_critic_probe_halfcheetah import (
    CriticProbeCallback,
    FixedUnifiedReturnCallback,
)

SAVE_DIR = "./npz_logs_alpha_halfcheetah"
STATE_DIR = "./npz_logs_step3_halfcheetah/state"
CONDITIONS = ["alpha_reset", "alpha_fixed"]
SPECIMENS = [("pure_sac", 8), ("fixed_prior", 8)]
# 戻し先: 健全シード（seed 7）の元ラン 100–500k 窓の α 平均
ALPHA_TARGETS = {"pure_sac": 0.1581, "fixed_prior": 0.1226}


def _state_zip_path(algo_type, seed, state_steps):
    return os.path.join(
        STATE_DIR,
        f"{algo_type}_halfcheetah_seed{seed}_state_{state_steps // 1000}k.zip",
    )


def _state_buffer_path(algo_type, seed, state_steps):
    return os.path.join(
        STATE_DIR,
        f"{algo_type}_halfcheetah_seed{seed}_state_{state_steps // 1000}k_replay_buffer.pkl",
    )


def _model_class(algo_type):
    if algo_type == "pure_sac":
        return PureSACWithLogging
    if algo_type == "fixed_prior":
        return SACWithFixedPrior
    raise ValueError(f"unknown algo_type: {algo_type}")


def _save_npz(file_path, model, eval_callback, probe_callback, log_starts):
    """ステップ3と同じキーの npz を保存する。"""
    def _slice(attr):
        values = list(getattr(model, attr))
        return np.array(values[log_starts.get(attr, 0):])

    save_dict = {
        "timesteps": np.array(eval_callback.timesteps),
        "returns": np.array(eval_callback.episode_returns),
        "pi_entropies": _slice("pi_entropies"),
        "alpha_logs": _slice("alpha_logs"),
        "probe_timesteps": np.array(probe_callback.probe_timesteps),
        "q1_mean": np.array(probe_callback.q1_mean),
        "q2_mean": np.array(probe_callback.q2_mean),
        "target_q_mean": np.array(probe_callback.target_q_mean),
        "next_log_prob_mean": np.array(probe_callback.next_log_prob_mean),
        "q_std": np.array(probe_callback.q_std),
        "td_abs_mean": np.array(probe_callback.td_abs_mean),
        "td_std": np.array(probe_callback.td_std),
        "critic_loss_probe": np.array(probe_callback.critic_loss_probe),
        "critic_grad_norm": np.array(probe_callback.critic_grad_norm),
        "actor_grad_norm": np.array(probe_callback.actor_grad_norm),
        "dormant_frac_qf0": np.array(probe_callback.dormant_frac_qf0),
        "dormant_frac_qf1": np.array(probe_callback.dormant_frac_qf1),
    }
    if hasattr(model, "kl_values"):
        save_dict["kl_values"] = _slice("kl_values")
    if hasattr(model, "beta_logs"):
        save_dict["beta_logs"] = _slice("beta_logs")
    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")
    print(f" probe points: {len(probe_callback.probe_timesteps)}")


def _current_alpha(model):
    try:
        return float(torch.exp(model.log_ent_coef.detach()).item())
    except Exception:
        return None


def _print_branch_state(model, algo_type, label):
    """分岐点の状態を表示する（読み込みと α 操作の確認用）。"""
    n_buf = None
    try:
        n_buf = int(model.replay_buffer.size())
    except Exception:
        pass
    msg = (f" [alpha] {label}: num_timesteps={model.num_timesteps:,}  "
           f"buffer={n_buf}  alpha={_current_alpha(model)}")
    if algo_type == "fixed_prior":
        beta = None
        try:
            beta = float(torch.exp(model.log_beta.detach()).item())
        except Exception:
            pass
        msg += f"  beta={beta}"
    print(msg)


def apply_alpha_condition(model, algo_type, condition):
    """α だけを操作する。他の部品には触らない。"""
    target = ALPHA_TARGETS[algo_type]
    log_target = math.log(target)
    with torch.no_grad():
        model.log_ent_coef.data.fill_(log_target)
    print(f" [alpha] log_ent_coef を log({target}) = {log_target:.4f} に設定"
          f"（適用後 alpha={_current_alpha(model)}）")

    if condition == "alpha_reset":
        # α 用オプティマイザの蓄積状態（Adam の m/v）を消す。
        # ステップ3でリセット対象のオプティマイザ状態を消した流儀と揃える。
        try:
            opt = model.ent_coef_optimizer
            removed = opt.state.pop(model.log_ent_coef, None)
            print(f" [alpha] ent_coef_optimizer の状態を消去"
                  f"（{'あり' if removed is not None else 'なし'}）")
        except Exception as e:
            print(f" [alpha] WARNING: オプティマイザ状態の消去に失敗: {e}")
    elif condition == "alpha_fixed":
        # 学習率を 0 にして α を固定する（更新式は走るが値は動かない）。
        for group in model.ent_coef_optimizer.param_groups:
            group["lr"] = 0.0
        lrs = [g["lr"] for g in model.ent_coef_optimizer.param_groups]
        print(f" [alpha] ent_coef_optimizer の学習率を 0 に設定: {lrs}")
    else:
        raise ValueError(f"unknown condition: {condition}")


def intervene(algo_type, seed, condition, state_steps=500_000,
              continue_steps=500_000, batch_size=256, probe_every=5000,
              tag=""):
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(SAVE_DIR, f"checkpoints_{condition}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    zip_path = _state_zip_path(algo_type, seed, state_steps)
    buffer_path = _state_buffer_path(algo_type, seed, state_steps)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"状態ファイルがありません: {zip_path}")
    if not os.path.exists(buffer_path):
        raise FileNotFoundError(f"バッファがありません: {buffer_path}")

    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    model_cls = _model_class(algo_type)
    if algo_type == "fixed_prior":
        # ステップ3と同じ読み込み専用サブクラス（先輩版クラスは無変更）
        class _LoadableFixedPrior(SACWithFixedPrior):
            def __init__(self, *args, beta_kl=0.01, beta_lr=1e-3,
                         target_kl=1.0, prior_std=2.0, **kwargs):
                super().__init__(*args, beta_kl=beta_kl, beta_lr=beta_lr,
                                 target_kl=target_kl, prior_std=prior_std,
                                 **kwargs)

        model = _LoadableFixedPrior.load(zip_path, env=train_env)
    else:
        model = model_cls.load(zip_path, env=train_env)
    model.load_replay_buffer(buffer_path)
    _print_branch_state(model, algo_type, f"読み込み直後（適用前, {condition}）")

    apply_alpha_condition(model, algo_type, condition)
    _print_branch_state(model, algo_type, f"条件適用後（{condition}）")

    log_starts = {
        attr: len(getattr(model, attr, []))
        for attr in ("pi_entropies", "alpha_logs", "kl_values", "beta_logs")
    }

    eval_callback = FixedUnifiedReturnCallback(
        eval_env=eval_env, eval_freq=5000, n_eval_episodes=5,
        deterministic=True, render=False,
    )
    probe_callback = CriticProbeCallback(
        algo_type=algo_type, seed=seed, save_dir=SAVE_DIR,
        checkpoint_dir=checkpoint_dir, batch_size=batch_size,
        probe_every=probe_every, checkpoint_every=100000,
    )

    print(f"[{algo_type.upper()}] ALPHA intervene Seed {seed} condition={condition} "
          f"({state_steps:,} -> {state_steps + continue_steps:,} steps)...")
    model.learn(total_timesteps=continue_steps, callback=[eval_callback, probe_callback],
                progress_bar=True, reset_num_timesteps=False)

    npz_path = os.path.join(
        SAVE_DIR, f"{algo_type}_halfcheetah_seed{seed}_{condition}{tag}.npz"
    )
    _save_npz(npz_path, model, eval_callback, probe_callback, log_starts=log_starts)

    train_env.close()
    eval_env.close()
    return npz_path


def run_plan_alpha(state_steps=500_000, continue_steps=500_000):
    for algo_type, seed in SPECIMENS:
        for condition in CONDITIONS:
            print(f"\n==========================================")
            print(f" ALPHA: {algo_type} | Seed: {seed} | condition: {condition}")
            print(f"==========================================")
            intervene(algo_type, seed, condition, state_steps=state_steps,
                      continue_steps=continue_steps)


def run_smoke():
    # ステップ3の試し回しで作った 10k 状態を使う（両手法とも保存済み）
    for algo_type, seed in SPECIMENS:
        for condition in CONDITIONS:
            print(f"\n=== ALPHA SMOKE: {algo_type} seed {seed} {condition} ===")
            intervene(algo_type, seed, condition, state_steps=10_000,
                      continue_steps=2_000, probe_every=1000, tag="_smoke")
    print("=== SMOKE DONE ===")


def main():
    parser = argparse.ArgumentParser(
        description="項目4: α を戻す介入（HalfCheetah-v5）"
    )
    parser.add_argument("--algo", choices=["pure_sac", "fixed_prior"], default="pure_sac")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--condition", choices=CONDITIONS, default="alpha_reset")
    parser.add_argument("--state-steps", type=int, default=500_000)
    parser.add_argument("--continue-steps", type=int, default=500_000)
    parser.add_argument("--intervene", action="store_true",
                        help="保存状態から α 条件を適用して継続する")
    parser.add_argument("--plan-alpha", action="store_true",
                        help="両手法×2条件の4本を順に回す（約6時間）")
    parser.add_argument("--smoke", action="store_true",
                        help="試し回し: 10k 状態 → 2条件 × 2k 継続（両手法）")
    args = parser.parse_args()

    print("=== 項目4: α を戻す介入 ===")
    print("注意: ほかの学習プロセスが動いている間は起動しないでください。")

    if args.smoke:
        run_smoke()
        return
    if args.plan_alpha:
        run_plan_alpha(state_steps=args.state_steps, continue_steps=args.continue_steps)
        return
    if args.intervene:
        intervene(args.algo, args.seed, args.condition,
                  state_steps=args.state_steps, continue_steps=args.continue_steps)
        return
    parser.error("--intervene / --plan-alpha / --smoke のいずれかを指定してください")


if __name__ == "__main__":
    main()
