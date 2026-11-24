#!/usr/bin/env python3
"""
フェーズ3 ステップ3: 500k 時点のリセット介入（因果の切り分け）
================================================================

位置づけ（2026-11-23 設計確定・ユーザー承認済み）
--------------------------------------------------
ステップ2（クリティック記録つき再走）の観察で、這い型に崩壊した
シードのクリティックは「壊れて」見えなかった（Q は実リターンと
桁が合い、TD 誤差は健全側より小さい）。観察では因果が決まらない
ため、ステップ3では固着した状態の部品を1つだけ入れ替えて継続し、
抜け出せるかどうかで固着の場所を切り分ける。

設計メモ: ステップ3設計_500kリセット介入_2026-11-23.md

標本と分岐点
------------
- 標本: seed 8 ペア（pure_sac / fixed_prior ともに這い型で崩壊）。
  ステップ2で再現済み（500k 評価が元ランと一致:
  pure_sac 2,130.8 / fixed_prior 1,883.8）。
- 分岐点: 500,000 steps。この時点で這い型は固定済みで、残り
  500k の継続で救済の有無を判定できる。

手順は2段に分かれる。

  第1段 (--make-state): seed 8 ペアを 0 から 500k まで、ステップ2と
  同一条件・同一プローブ記録で再走し、500k 時点の「モデル＋
  リプレイバッファ」を保存する。ステップ2のチェックポイントには
  バッファが含まれないため、この作り直しが必要。全条件はこの
  1つの保存状態から分岐し、開始点が完全一致する。

  第2段 (--intervene): 保存状態を読み込み、条件を1つだけ適用して
  1M まで継続する。

条件（標本ごとに4条件）
------------------------
- control : 何も変えずに継続。「保存状態から再開する」操作自体の
            影響を代表する対照。
- critic  : クリティック（Qネットワーク）だけを初期化し直す。
            目標側クリティックは初期化したクリティックのコピーに
            する（通常の初期化と同じ）。アクター（方策）と
            バッファは保持。
- buffer  : リプレイバッファだけを空にする（読み込まない）。
            ネットワークは両方とも保持し、データは空から貯め直す。
- actor   : アクター（方策ネットワーク）だけを初期化し直す。
            クリティックとバッファは保持。

単一変数の線引き（先に固定）
----------------------------
- actor リセットは方策ネットワークの重みだけが対象。α（SAC の
  エントロピー温度）と、先輩版の β はスカラーの調整量であり、
  ネットワークではないので保持する。
- critic / actor リセットでは、対象モジュールに付いたオプティ
  マイザ（この SB3 ではアクターとクリティックに別々の Adam）の
  蓄積状態を、対象パラメータについて消す（ベストエフォート。
  内部構造が想定と違えば警告だけ出して重みの初期化のみで進む）。
  ※初版は「方策全体で1つの共有 Adam」と想定して model.policy.
  optimizer を見に行き、試し回しで属性が無いことが判明したため、
  モジュール直付けの optimizer を見るよう修正した（2026-11-23）。
- 重みの初期化は各 Linear 層の reset_parameters() で行う
  （新規構築時と同じ分布からの初期化）。

読み分け（設計メモと同じ）
--------------------------
- critic で救済 → 価値関数の側の固着
- buffer で救済 → データの循環
- actor で救済 → 方策側の固着
- どれでも救済されない → 局所最適の読みが強まる
- SAC と先輩版で結果が違えば、ADR のカバー範囲の材料になる

判定基準（実行前に固定）
------------------------
継続の最終区間（900k–1M）の評価リターン平均で判定し、既存の
崩壊の定義（4,000 未満＝崩壊）をそのまま使う。4,000 以上に
抜けたら救済。

正直な留保
----------
- 2026-11-24 の本番で、先輩版の第2段1本目（control）の読み込みが
  TypeError で落ち、バッチが止まった。原因は SACWithFixedPrior の
  必須引数（beta_kl 等）を SB3 の load が再構築時に渡さないこと。
  load への追加引数の明示渡しは効かなかった（SB3 はそれを
  コンストラクタではなく __dict__ の更新に使う）ため、4つの設定に
  初期値を持たせた読み込み専用サブクラス経由に切り替えた。
  試し回しにも先輩版の読み込みを含めた。SAC 側の第2段4本は
  修正前に完走済みで、先輩版の第1段（500k 状態）も保存済みの
  ため、再開は先輩版の第2段4本だけでよい。
- SB3 の保存・読み込みでは、一部の最適化器の状態（先輩版の
  β 用オプティマイザなど、モデル直付けのもの）は保存されない
  ことがある。これは全条件に共通で、control 条件がその影響を
  代表する。分岐直後の α・β・バッファ件数をログに出すので、
  読み込みが正しく行われたかはそこで確認できる。
- 作り直した 500k 状態は再走の産物であり、元ランのビット単位の
  複製ではない。500k 評価がステップ2の再現値（2,130.8 /
  1,883.8）と近いことで「崩壊した標本である」ことを確認する。

条件（ステップ2・既存ランと同一）
----------------------------------
- 環境: HalfCheetah-v5 / batch_size=256
- pure_sac   : PureSACWithLogging（sac_models.py）
- fixed_prior: SACWithFixedPrior（sac_adr_main.py、先輩本人の実装）
  prior_std=2.0 / target_kl=1.0 / beta_kl=0.01 / beta_lr=1e-3
- 評価: 5,000 steps ごとに 5 エピソード（決定論的）。
  train seed = seed、eval seed = seed+1000。
- プローブ記録: ステップ2の CriticProbeCallback をそのまま使う
  （run_critic_probe_halfcheetah.py から import）。

使い方（keisanB の rl-motion 直下で）
-------------------------------------
# 試し回し（pure_sac seed 8: 10k で状態保存 → 4条件 × 2k 継続）
python3 run_step3_intervention_halfcheetah.py --smoke

# 第1段: 500k 状態の作り直し（2本）
python3 run_step3_intervention_halfcheetah.py --make-state --algo pure_sac --seed 8
python3 run_step3_intervention_halfcheetah.py --make-state --algo fixed_prior --seed 8

# 第2段: 介入1本
python3 run_step3_intervention_halfcheetah.py --intervene --algo pure_sac --seed 8 --condition critic

# 全部を順に（第1段2本＋第2段8本。約2日の見込み）
python3 run_step3_intervention_halfcheetah.py --plan

出力
----
保存先は ./npz_logs_step3_halfcheetah/。
- 状態: state/{algo}_halfcheetah_seed{seed}_state_{N}k.zip と
  state/{algo}_halfcheetah_seed{seed}_state_{N}k_replay_buffer.pkl
- 第1段の npz: {algo}_halfcheetah_seed{seed}_step3state.npz
  （ステップ2と同じキー。500k 評価の再現確認用）
- 第2段の npz: {algo}_halfcheetah_seed{seed}_step3_{condition}.npz
  （ステップ2と同じキー。timesteps は 500k から 1M の絶対値）
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior
from run_critic_probe_halfcheetah import (
    CriticProbeCallback,
    FixedUnifiedReturnCallback,
)

SAVE_DIR = "./npz_logs_step3_halfcheetah"
STATE_DIR = os.path.join(SAVE_DIR, "state")
CONDITIONS = ["control", "critic", "buffer", "actor"]
SPECIMENS = [("pure_sac", 8), ("fixed_prior", 8)]


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


def _build_model(algo_type, train_env, seed, batch_size=256):
    """既存ランと同一のモデル構築（ステップ2の run_experiment と同じ）。"""
    if algo_type == "pure_sac":
        return PureSACWithLogging(
            "MlpPolicy", train_env, batch_size=batch_size, seed=seed, verbose=0
        )
    elif algo_type == "fixed_prior":
        return SACWithFixedPrior(
            "MlpPolicy",
            train_env,
            batch_size=batch_size,
            beta_kl=0.01,
            beta_lr=1e-3,
            target_kl=1.0,
            prior_std=2.0,
            seed=seed,
            verbose=0,
        )
    raise ValueError(f"unknown algo_type: {algo_type}")


def _model_class(algo_type):
    if algo_type == "pure_sac":
        return PureSACWithLogging
    if algo_type == "fixed_prior":
        return SACWithFixedPrior
    raise ValueError(f"unknown algo_type: {algo_type}")


def _save_npz(file_path, model, eval_callback, probe_callback, log_starts):
    """ステップ2と同じキーの npz を保存する。

    log_starts はモデルの記録リストごとの開始位置（辞書）で、
    継続分だけを切り出すために使う（第1段では全リスト 0）。
    """
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


def _print_branch_state(model, algo_type, label):
    """分岐点の状態を表示する（読み込みが正しいかの確認用）。"""
    n_buf = None
    try:
        n_buf = int(model.replay_buffer.size())
    except Exception:
        pass
    alpha = None
    try:
        alpha = float(torch.exp(model.log_ent_coef.detach()).item())
    except Exception:
        pass
    msg = f" [step3] {label}: num_timesteps={model.num_timesteps:,}  buffer={n_buf}  alpha={alpha}"
    if algo_type == "fixed_prior":
        beta = None
        try:
            beta = float(torch.exp(model.log_beta.detach()).item())
        except Exception:
            pass
        msg += f"  beta={beta}"
    print(msg)


def _clear_optimizer_state_for(module, model):
    """対象モジュールのオプティマイザの蓄積状態のうち、そのモジュールの
    パラメータ分だけを消す。

    この SB3 ではオプティマイザはアクター／クリティックの各モジュールに
    直付けされている（model.policy.optimizer は無い。2026-11-23 の
    試し回しで確認）。念のため方策側の optimizer もフォールバックで見る。
    内部構造が想定と違えば警告だけ出して続行する（重みの初期化が
    介入の本体）。
    """
    try:
        optimizer = getattr(module, "optimizer", None)
        if optimizer is None:
            optimizer = getattr(model.policy, "optimizer", None)
        if optimizer is None:
            raise AttributeError("対象モジュールの optimizer が見つかりません")
        params = list(module.parameters())
        removed = 0
        for p in params:
            if p in optimizer.state:
                del optimizer.state[p]
                removed += 1
        print(f" [step3] オプティマイザ状態を消去: {removed}/{len(params)} パラメータ")
    except Exception as e:
        print(f" [step3] WARNING: オプティマイザ状態の消去に失敗（重みの初期化のみで進みます）: {e}")


def _reset_module_weights(module):
    """モジュール内の全 Linear 層を初期化し直す。"""
    n = 0
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()
            n += 1
    return n


def apply_condition(model, algo_type, condition):
    """保存状態のモデルに、条件を1つだけ適用する。"""
    if condition == "control":
        print(" [step3] 条件: control（何も変えずに継続）")
    elif condition == "critic":
        n = _reset_module_weights(model.critic)
        model.critic_target.load_state_dict(model.critic.state_dict())
        _clear_optimizer_state_for(model.critic, model)
        print(f" [step3] 条件: critic（Qネットワーク {n} 層を初期化、目標側も同期）")
    elif condition == "actor":
        n = _reset_module_weights(model.actor)
        _clear_optimizer_state_for(model.actor, model)
        print(f" [step3] 条件: actor（方策ネットワーク {n} 層を初期化。α・β は保持）")
    elif condition == "buffer":
        # バッファは読み込まずに開始するため、ここでは何もしない
        print(" [step3] 条件: buffer（バッファは空から開始）")
    else:
        raise ValueError(f"unknown condition: {condition}")


# ---------------------------------------------------------------------------
# 第1段: 500k 状態の作り直し
# ---------------------------------------------------------------------------
def make_state(algo_type, seed, state_steps=500_000, batch_size=256,
               probe_every=5000, tag=""):
    os.makedirs(STATE_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(SAVE_DIR, "checkpoints_state")
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    eval_callback = FixedUnifiedReturnCallback(
        eval_env=eval_env, eval_freq=5000, n_eval_episodes=5,
        deterministic=True, render=False,
    )
    probe_callback = CriticProbeCallback(
        algo_type=algo_type, seed=seed, save_dir=SAVE_DIR,
        checkpoint_dir=checkpoint_dir, batch_size=batch_size,
        probe_every=probe_every, checkpoint_every=100000,
    )
    model = _build_model(algo_type, train_env, seed, batch_size=batch_size)

    print(f"[{algo_type.upper()}] STEP3 make-state Seed {seed} "
          f"(0 -> {state_steps:,} steps)...")
    model.learn(total_timesteps=state_steps,
                callback=[eval_callback, probe_callback], progress_bar=True)

    zip_path = _state_zip_path(algo_type, seed, state_steps)
    buffer_path = _state_buffer_path(algo_type, seed, state_steps)
    model.save(zip_path)
    print(f" [step3] model saved: {zip_path}")
    try:
        model.save_replay_buffer(buffer_path)
        print(f" [step3] replay buffer saved: {buffer_path}")
    except Exception as e:
        train_env.close()
        eval_env.close()
        raise RuntimeError(f"バッファの保存に失敗しました: {e}")

    _print_branch_state(model, algo_type, "make-state 終了時点")

    npz_path = os.path.join(
        SAVE_DIR, f"{algo_type}_halfcheetah_seed{seed}_step3state{tag}.npz"
    )
    _save_npz(npz_path, model, eval_callback, probe_callback, log_starts={})

    train_env.close()
    eval_env.close()
    return zip_path, buffer_path


# ---------------------------------------------------------------------------
# 第2段: 介入して継続
# ---------------------------------------------------------------------------
def intervene(algo_type, seed, condition, state_steps=500_000,
              continue_steps=500_000, batch_size=256, probe_every=5000,
              tag=""):
    os.makedirs(SAVE_DIR, exist_ok=True)
    checkpoint_dir = os.path.join(SAVE_DIR, f"checkpoints_{condition}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    zip_path = _state_zip_path(algo_type, seed, state_steps)
    buffer_path = _state_buffer_path(algo_type, seed, state_steps)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"状態ファイルがありません: {zip_path}（先に --make-state を実行してください）")
    if condition != "buffer" and not os.path.exists(buffer_path):
        raise FileNotFoundError(f"バッファがありません: {buffer_path}（先に --make-state を実行してください）")

    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    model_cls = _model_class(algo_type)
    if algo_type == "fixed_prior":
        # 先輩版 SACWithFixedPrior は beta_kl / beta_lr / target_kl / prior_std
        # が必須の引数（デフォルトなし）で、SB3 の load は再構築のときに
        # これらを渡さないため TypeError になる（2026-11-24 の本番で判明）。
        # load に追加引数を渡しても、それはコンストラクタではなく読み込み
        # 後の __dict__ 更新に使われるだけなので効かない（同日の試し回しで
        # 確認）。そこで、4つの設定に構築時と同じ初期値を持たせた読み込み
        # 専用のサブクラス経由で読み込む。先輩版クラス自体は無変更で、
        # 保存済みの重みと log_beta は SB3 の load が復元する。復元が
        # 正しいかは分岐点の表示（β の値）で確認する。
        class _LoadableFixedPrior(SACWithFixedPrior):
            def __init__(self, *args, beta_kl=0.01, beta_lr=1e-3,
                         target_kl=1.0, prior_std=2.0, **kwargs):
                super().__init__(*args, beta_kl=beta_kl, beta_lr=beta_lr,
                                 target_kl=target_kl, prior_std=prior_std,
                                 **kwargs)

        model = _LoadableFixedPrior.load(zip_path, env=train_env)
    else:
        model = model_cls.load(zip_path, env=train_env)
    if condition != "buffer":
        model.load_replay_buffer(buffer_path)
    _print_branch_state(model, algo_type, f"読み込み直後（適用前, {condition}）")

    apply_condition(model, algo_type, condition)
    _print_branch_state(model, algo_type, f"条件適用後（{condition}）")

    # モデルの記録リストは継続分だけを切り出す（リストごとに開始位置を持つ）
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

    print(f"[{algo_type.upper()}] STEP3 intervene Seed {seed} condition={condition} "
          f"({state_steps:,} -> {state_steps + continue_steps:,} steps)...")
    model.learn(total_timesteps=continue_steps, callback=[eval_callback, probe_callback],
                progress_bar=True, reset_num_timesteps=False)

    npz_path = os.path.join(
        SAVE_DIR, f"{algo_type}_halfcheetah_seed{seed}_step3_{condition}{tag}.npz"
    )
    _save_npz(npz_path, model, eval_callback, probe_callback, log_starts=log_starts)

    train_env.close()
    eval_env.close()
    return npz_path


def run_plan(state_steps=500_000, continue_steps=500_000):
    for algo_type, seed in SPECIMENS:
        make_state(algo_type, seed, state_steps=state_steps)
    for algo_type, seed in SPECIMENS:
        for condition in CONDITIONS:
            print(f"\n==========================================")
            print(f" STEP3: {algo_type} | Seed: {seed} | condition: {condition}")
            print(f"==========================================")
            intervene(algo_type, seed, condition, state_steps=state_steps,
                      continue_steps=continue_steps)


def run_smoke():
    print("=== STEP3 SMOKE (pure_sac seed 8: 10k state -> 4 conditions x 2k) ===")
    make_state("pure_sac", 8, state_steps=10_000, probe_every=1000, tag="_smoke")
    for condition in CONDITIONS:
        intervene("pure_sac", 8, condition, state_steps=10_000,
                  continue_steps=2_000, probe_every=1000, tag="_smoke")
    # 先輩版は load の経路が SAC と違う（必須の引数がある）ため、状態保存と
    # 読み込み（control だけ）も試し回しに含める（2026-11-24 追加。
    # 初版の試し回しは SAC のみで、本番の先輩版1本目で落ちた反省から）
    print("=== STEP3 SMOKE (fixed_prior seed 8: 10k state -> control x 2k) ===")
    make_state("fixed_prior", 8, state_steps=10_000, probe_every=1000, tag="_smoke")
    intervene("fixed_prior", 8, "control", state_steps=10_000,
              continue_steps=2_000, probe_every=1000, tag="_smoke")
    print("=== SMOKE DONE ===")


def main():
    parser = argparse.ArgumentParser(
        description="フェーズ3 ステップ3: 500k 時点のリセット介入（HalfCheetah-v5）"
    )
    parser.add_argument("--algo", choices=["pure_sac", "fixed_prior"], default="pure_sac")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--condition", choices=CONDITIONS, default="control")
    parser.add_argument("--state-steps", type=int, default=500_000)
    parser.add_argument("--continue-steps", type=int, default=500_000)
    parser.add_argument("--make-state", action="store_true",
                        help="第1段: 500k 状態（モデル＋バッファ）を作る")
    parser.add_argument("--intervene", action="store_true",
                        help="第2段: 保存状態から条件を適用して継続する")
    parser.add_argument("--plan", action="store_true",
                        help="第1段2本＋第2段8本を順に回す")
    parser.add_argument("--smoke", action="store_true",
                        help="試し回し: 10k で状態保存 → 4条件 × 2k 継続")
    args = parser.parse_args()

    print("=== フェーズ3 ステップ3: 500k 時点のリセット介入 ===")
    print("注意: ほかの学習プロセスが動いている間は起動しないでください。")

    if args.smoke:
        run_smoke()
        return
    if args.plan:
        run_plan(state_steps=args.state_steps, continue_steps=args.continue_steps)
        return
    if args.make_state:
        make_state(args.algo, args.seed, state_steps=args.state_steps)
        return
    if args.intervene:
        intervene(args.algo, args.seed, args.condition,
                  state_steps=args.state_steps, continue_steps=args.continue_steps)
        return
    parser.error("--make-state / --intervene / --plan / --smoke のいずれかを指定してください")


if __name__ == "__main__":
    main()
