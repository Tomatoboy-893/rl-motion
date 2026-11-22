#!/usr/bin/env python3
"""
フェーズ3 ステップ2: クリティック記録つき再走（観察）
====================================================

位置づけ（2026-11-21 承認済み・2026-11-22 作成）
----------------------------------------------
2026-11-21 の npz 診断で、分布側の量（エントロピー・KL・β）は這い型失敗の
決定要因から外れた。残る容疑は (1) クリティックの価値学習の失敗、
(2) リプレイバッファのデータ循環、(3) 這い歩行自体が局所最適、の3つ。
このスクリプトは (1) を観察で切り分けるための再走用で、崩壊シードと
健全シードを「条件は同じまま、記録だけ足して」走らせ直す。

対象（計画書「今後の流れ_2026-11-18.md」フェーズ3 ステップ2、2026-11-22 統一版）
------------------------------------------------------------------
- 第1陣: seed 8 ペア（SAC seed 8 と先輩版 seed 8。初期σがほぼ同一で
  両方とも這い型に崩壊した組）＋対照の SAC seed 7（健全）
- 尺度: 分岐は 500k までにほぼ決まるため 500k〜1M steps。
  既定は 1,000,000 steps（1本あたり約1.5〜3時間の見込み）。
  --total-steps で変更できる。

条件（既存ランと完全に同一。アルゴリズムには一切手を入れない）
------------------------------------------------------------------
- 環境: HalfCheetah-v5 / batch_size=256
- pure_sac   : PureSACWithLogging（sac_models.py、素の SAC）
- fixed_prior: SACWithFixedPrior（sac_adr_main.py、先輩本人の実装）
  prior_std=2.0 / target_kl=1.0（次元あたり）/ beta_kl=0.01 / beta_lr=1e-3
- 評価: 5,000 steps ごとに 5 エピソード（決定論的方策）。
  評価コールバックは拡張ラン（run_halfcheetah_10seed_extension.py）
  と同じ FixedUnifiedReturnCallback を使う。
- 環境シードも拡張ランと同じ（train: seed、eval: seed+1000）。

追加する記録（これだけが既存ランとの違い）
------------------------------------------------------------------
5,000 steps ごとに、学習中のリプレイバッファから batch_size=256 の
バッチを1つ取り、現在のクリティックで次を測る（推定用の読み取りで、
学習の計算そのものには触れない）。

- q1_mean / q2_mean: バッチ上の Q1・Q2 の平均
- target_q_mean: TD ターゲットの平均
- q_std: バッチ上の Q1 の標準偏差（Q のばらつき）
- td_abs_mean / td_std: TD 誤差（ターゲット − Q1）の絶対値平均と標準偏差
- critic_loss_probe: MSE(Q1, ターゲット) + MSE(Q2, ターゲット)。
  sac_models.py の train() と同じ式（0.5 倍しない「和」の形）で計算する。
- critic_grad_norm / actor_grad_norm: 直前の学習ステップの勾配ノルム
  （コールバック時点に残っている p.grad から読む。学習直後の値ではなく
  「直近の学習ブロックの最後の1ステップ」の値である点に注意）
- dormant_frac_qf0 / dormant_frac_qf1: クリティック隠れ層の休眠率の目安。
  プローブ時の順伝播で、バッチ平均の |活性化| が 1e-3 未満のユニットの割合。
  SB3 の Critic の内部構造が想定と違う場合は NaN を入れて警告だけ出す
  （ベストエフォート。失敗しても学習は止めない）。

あわせて 100k steps ごとにモデルをチェックポイント保存する
（./npz_logs_critic_probe_halfcheetah/checkpoints/）。
現行実験はモデル未保存で事後調査ができなかった反省を受けた措置で、
以後の標本は後からでもクリティックを調べられる。
注意: SB3 の model.save() はリプレイバッファを含まない。ステップ3の
「チェックポイントから再開してリセット介入」にはバッファの扱いを別途
決める必要がある（このスクリプトでは解決していない。要検討事項）。

読み分けの目安（計画書と同じ）
--------------------------------
- Q が膨らみ続ける → 過大評価・発散型（Hussing et al. 2024,
  https://arxiv.org/abs/2403.05996）
- Q は安定だが序盤データに張り付いて動かない（TD 誤差が下がらない、
  Q のばらつきが小さい） → primacy 型（Nikishin et al. 2022,
  https://arxiv.org/abs/2205.07802）
- Q が正常で、這い歩行を近傍の最良として妥当に評価している
  → 局所最適型（クリティックは「悪くない」）
- 健全シード（seed 7）との並べ比べが前提。1本だけでは判定しない。

学習への影響を出さないための仕組み（重要）
--------------------------------------------
プローブは2つの乱数源を使うため、学習の乱数列を乱さないよう
スナップショット→復元の保護を入れている。

1. replay_buffer.sample() は NumPy の乱数を使う。
2. actor.action_log_prob() は方策のサンプリングで torch の乱数を使う。
   （ターゲット Q の計算に次行動のサンプルが必要なため）

プローブの前後で NumPy / Python random / torch（CPU・CUDA）/
リプレイバッファ内部の乱数（np_random 属性がある場合）の状態を保存し、
プローブ後にすべて復元する。復元が壊れていないかは、プローブごとに
NumPy の乱数状態が保存前と一致することを assert で確認する
（不一致なら例外で止める。静かに学習を変えるより止まる方がよい）。
またプローブの順伝播は torch.no_grad() で行い、勾配には触れない。
チェックポイント保存（model.save）は乱数を使わない。

それでも「再走は元のランと完全同一ではない」点は正直に書いておく。
プローブ自体は学習の計算に触れない設計だが、再走という行為自体が
元のランとは別の実行であり、同一シードでも計算機の状態などで
数値がわずかにずれる可能性はある。比較は「同じスクリプト・同じ条件で
走らせ直した崩壊シードと健全シードの対比」として読むこと。

使い方（keisanB の rl-motion 直下で）
---------------------------------------
# まず試し回し（10,000 steps。記録キー・チェックポイント・乱数復元の確認）
python3 run_critic_probe_halfcheetah.py --algo pure_sac --seed 8 --smoke

# 本番1本（既定 1M steps）
python3 run_critic_probe_halfcheetah.py --algo pure_sac --seed 8

# 計画の3本を順に（SAC seed 8 → SAC seed 7 → 先輩版 seed 8）
python3 run_critic_probe_halfcheetah.py --plan

# 500k だけ、などの指定
python3 run_critic_probe_halfcheetah.py --algo pure_sac --seed 7 --total-steps 500000

実行前の確認
------------
- 10シード拡張ランが終わっていること（計算機を共有するため。
  ほかの学習プロセスが動いている間は起動しないこと）。
- 保存先は ./npz_logs_critic_probe_halfcheetah/ で、既存の npz を
  上書きしない。ファイル名に _criticprobe を付ける。

出力
----
- npz: ./npz_logs_critic_probe_halfcheetah/{algo}_halfcheetah_seed{seed}_criticprobe.npz
  既存キー（timesteps / returns / pi_entropies / alpha_logs /
  先輩版のみ kl_values / beta_logs）に加え、プローブのキー
  （probe_timesteps / q1_mean / q2_mean / target_q_mean / q_std /
  td_abs_mean / td_std / critic_loss_probe / critic_grad_norm /
  actor_grad_norm / dormant_frac_qf0 / dormant_frac_qf1 /
  next_log_prob_mean）を入れる。next_log_prob_mean は α の更新が使う
  サンプルされた log π のバッチ平均で、記録上のエントロピーとの
  ずれを見るために 2026-11-22 に追加した。
- チェックポイント: ./npz_logs_critic_probe_halfcheetah/checkpoints/
  {algo}_halfcheetah_seed{seed}_criticprobe_{N}k.zip（100k ごと＋最終）
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from sac_models import PureSACWithLogging
from sac_adr_main import SACWithFixedPrior


# ---------------------------------------------------------------------------
# 評価コールバック（拡張ランと同一）
# ---------------------------------------------------------------------------
class FixedUnifiedReturnCallback(EvalCallback):
    """評価周期（eval_freq）のタイミングでのみ重複なくデータを保存する修正コールバック
    （run_halfcheetah_experiments.py / run_halfcheetah_10seed_extension.py と同じもの）"""

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


# ---------------------------------------------------------------------------
# 乱数状態のスナップショット / 復元
# ---------------------------------------------------------------------------
def _snapshot_rng(model):
    """プローブが使う乱数源の状態をすべて保存する。"""
    state = {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "buffer": None,
        "buffer_kind": None,
    }
    buf = getattr(model, "replay_buffer", None)
    rng = getattr(buf, "np_random", None) if buf is not None else None
    if rng is not None:
        try:
            state["buffer"] = rng.get_state()
            state["buffer_kind"] = "random_state"
        except Exception:
            try:
                # numpy Generator の場合
                state["buffer"] = rng.bit_generator.state
                state["buffer_kind"] = "generator"
            except Exception:
                state["buffer"] = None
    return state


def _restore_rng(state, model):
    """スナップショットの状態へすべて戻す。"""
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["buffer"] is not None:
        buf = getattr(model, "replay_buffer", None)
        rng = getattr(buf, "np_random", None) if buf is not None else None
        if rng is not None:
            try:
                if state["buffer_kind"] == "random_state":
                    rng.set_state(state["buffer"])
                elif state["buffer_kind"] == "generator":
                    rng.bit_generator.state = state["buffer"]
            except Exception:
                pass


def _check_rng_restored(state):
    """NumPy の乱数状態が保存前と一致することを確認する（復元の自己検査）。"""
    now = np.random.get_state()
    before = state["numpy"]
    if before[0] != now[0] or before[1].tobytes() != now[1].tobytes() or before[2] != now[2]:
        raise RuntimeError(
            "[critic-probe] 乱数状態の復元に失敗しました。学習の乱数列が変わる "
            "おそれがあるため停止します。"
        )


def _grad_norm(params):
    """p.grad が残っているパラメータの勾配ノルム（L2）。勾配が無ければ NaN。"""
    total = 0.0
    found = False
    for p in params:
        if p.grad is not None:
            total += float(p.grad.detach().norm(2).item()) ** 2
            found = True
    return float(np.sqrt(total)) if found else float("nan")


# ---------------------------------------------------------------------------
# クリティック・プローブのコールバック
# ---------------------------------------------------------------------------
class CriticProbeCallback(BaseCallback):
    """5,000 steps ごとにクリティックの状態を測り、100k ごとに保存する。

    測るのは「その時点のバッファから取った1バッチに対する現在の
    クリティックの値」で、学習の計算そのものには触れない。
    """

    def __init__(
        self,
        algo_type,
        seed,
        save_dir,
        checkpoint_dir,
        batch_size=256,
        probe_every=5000,
        checkpoint_every=100000,
        verbose=0,
    ):
        super().__init__(verbose)
        self.algo_type = algo_type
        self.seed = seed
        self.save_dir = save_dir
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.probe_every = probe_every
        self.checkpoint_every = checkpoint_every

        self.probe_timesteps = []
        self.q1_mean = []
        self.q2_mean = []
        self.target_q_mean = []
        self.next_log_prob_mean = []
        self.q_std = []
        self.td_abs_mean = []
        self.td_std = []
        self.critic_loss_probe = []
        self.critic_grad_norm = []
        self.actor_grad_norm = []
        self.dormant_frac_qf0 = []
        self.dormant_frac_qf1 = []

        # 休眠率の計測用フック（ReLU の出力だけを、計測中のみ拾う）
        # model は _on_training_start で確定するため、フックの装着もそこで行う。
        self._capturing = False
        self._activations = {}
        self._dormant_ok = None  # 一度でも取れれば True、構造違いなら False
        self._dormant_warned = False

    def _attach_hooks_to_critic(self, critic):
        for name, net in (("qf0", getattr(critic, "qf0", None)),
                          ("qf1", getattr(critic, "qf1", None))):
            if net is None:
                continue
            for module in net.modules():
                if isinstance(module, nn.ReLU):
                    module.register_forward_hook(self._make_hook(name))

    def _make_hook(self, net_name):
        def hook(module, inputs, output):
            if self._capturing:
                self._activations.setdefault(net_name, []).append(
                    output.detach().abs().mean(dim=0)
                )
        return hook

    def _dormant_fraction(self, net_name):
        """フックで集めた活性化から休眠率を出す。取れなければ NaN。"""
        acts = self._activations.get(net_name)
        if not acts:
            return float("nan")
        # バッチ平均の |活性化|（ユニットごと）を全フック分つなげる
        per_unit = torch.cat([a.flatten() for a in acts])
        if per_unit.numel() == 0:
            return float("nan")
        return float((per_unit < 1e-3).float().mean().item())

    # -- SB3 コールバック ---------------------------------------------------
    def _on_training_start(self) -> None:
        # model が確定するこのタイミングでフックを付ける（init 時は model が無いため）
        try:
            self._attach_hooks_to_critic(self.model.critic)
            self._dormant_ok = True
        except Exception as e:
            if not self._dormant_warned:
                print(f" [critic-probe] 休眠率フックを付けられませんでした（NaN で記録）: {e}")
                self._dormant_warned = True
            self._dormant_ok = False

    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if steps > 0 and steps % self.probe_every == 0:
            self._probe()
        if steps > 0 and steps % self.checkpoint_every == 0:
            self._save_checkpoint(steps)
        return True

    def _on_training_end(self) -> None:
        # 最終チェックポイント（100k の倍数でなければここで保存）
        if self.num_timesteps % self.checkpoint_every != 0:
            self._save_checkpoint(self.num_timesteps)

    # -- チェックポイント ----------------------------------------------------
    def _save_checkpoint(self, steps):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.checkpoint_dir,
            f"{self.algo_type}_halfcheetah_seed{self.seed}_criticprobe_{steps // 1000}k.zip",
        )
        try:
            self.model.save(path)
            print(f" [critic-probe] checkpoint saved: {path}")
        except Exception as e:
            print(f" [critic-probe] WARNING: チェックポイント保存に失敗: {e}")

    # -- プローブ本体 --------------------------------------------------------
    def _probe(self):
        model = self.model
        steps = self.num_timesteps
        try:
            if model.replay_buffer.size() < self.batch_size:
                print(f" [critic-probe] {steps}: バッファ不足のためスキップ")
                return
        except Exception:
            pass  # size() が無ければそのまま進む

        rng_state = _snapshot_rng(model)
        try:
            with torch.no_grad():
                replay_data = model.replay_buffer.sample(self.batch_size)
                obs = replay_data.observations
                actions = replay_data.actions
                next_obs = replay_data.next_observations
                rewards = replay_data.rewards
                dones = replay_data.dones

                # TD ターゲット（sac_models.py の train() と同じ式）
                next_actions, next_log_prob = model.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.unsqueeze(-1)
                next_q = torch.min(*model.critic_target(next_obs, next_actions))
                alpha = torch.exp(model.log_ent_coef.detach())
                target_q = rewards + (1 - dones) * model.gamma * (next_q - alpha * next_log_prob)

                # 休眠率の計測中だけフックを有効にする
                self._capturing = True
                self._activations = {}
                try:
                    current_q1, current_q2 = model.critic(obs, actions)
                finally:
                    self._capturing = False

                td_error = target_q - current_q1
                critic_loss = (
                    torch.nn.functional.mse_loss(current_q1, target_q)
                    + torch.nn.functional.mse_loss(current_q2, target_q)
                )

                row = {
                    "q1_mean": float(current_q1.mean().item()),
                    "q2_mean": float(current_q2.mean().item()),
                    "target_q_mean": float(target_q.mean().item()),
                    "next_log_prob_mean": float(next_log_prob.mean().item()),
                    "q_std": float(current_q1.std().item()),
                    "td_abs_mean": float(td_error.abs().mean().item()),
                    "td_std": float(td_error.std().item()),
                    "critic_loss": float(critic_loss.item()),
                    "dormant_qf0": self._dormant_fraction("qf0"),
                    "dormant_qf1": self._dormant_fraction("qf1"),
                }
            # 勾配ノルム（直前の学習ステップの残り。no_grad の外で読む）
            row["critic_grad_norm"] = _grad_norm(model.critic.parameters())
            row["actor_grad_norm"] = _grad_norm(model.actor.parameters())
        finally:
            _restore_rng(rng_state, model)
            _check_rng_restored(rng_state)

        self.probe_timesteps.append(steps)
        self.q1_mean.append(row["q1_mean"])
        self.q2_mean.append(row["q2_mean"])
        self.target_q_mean.append(row["target_q_mean"])
        self.next_log_prob_mean.append(row["next_log_prob_mean"])
        self.q_std.append(row["q_std"])
        self.td_abs_mean.append(row["td_abs_mean"])
        self.td_std.append(row["td_std"])
        self.critic_loss_probe.append(row["critic_loss"])
        self.critic_grad_norm.append(row["critic_grad_norm"])
        self.actor_grad_norm.append(row["actor_grad_norm"])
        self.dormant_frac_qf0.append(row["dormant_qf0"])
        self.dormant_frac_qf1.append(row["dormant_qf1"])
        print(
            f" [critic-probe] {steps:>7,}: "
            f"Q1 {row['q1_mean']:8.1f}  targetQ {row['target_q_mean']:8.1f}  "
            f"|TD| {row['td_abs_mean']:7.2f}  critic_loss {row['critic_loss']:8.2f}  "
            f"Qstd {row['q_std']:6.2f}  gradC {row['critic_grad_norm']:.2e}"
        )


# ---------------------------------------------------------------------------
# 1本の実行（拡張ランと同一条件＋プローブ）
# ---------------------------------------------------------------------------
def run_experiment(algo_type, seed, total_timesteps=1_000_000, batch_size=256,
                   probe_every=5000, checkpoint_every=100000):
    save_dir = "./npz_logs_critic_probe_halfcheetah"
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_env = make_vec_env("HalfCheetah-v5", n_envs=1, seed=seed)
    eval_env = gym.make("HalfCheetah-v5")
    eval_env.reset(seed=seed + 1000)

    eval_callback = FixedUnifiedReturnCallback(
        eval_env=eval_env,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )
    probe_callback = CriticProbeCallback(
        algo_type=algo_type,
        seed=seed,
        save_dir=save_dir,
        checkpoint_dir=checkpoint_dir,
        batch_size=batch_size,
        probe_every=probe_every,
        checkpoint_every=checkpoint_every,
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
            beta_kl=0.01,
            beta_lr=1e-3,
            target_kl=1.0,
            prior_std=2.0,
            seed=seed,
            verbose=0,
        )
    else:
        raise ValueError(f"unknown algo_type: {algo_type}")

    print(f"[{algo_type.upper()}] Starting Seed {seed} "
          f"(critic-probe, {total_timesteps:,} steps, batch_size={batch_size})...")
    model.learn(total_timesteps=total_timesteps,
                callback=[eval_callback, probe_callback], progress_bar=True)

    train_env.close()
    eval_env.close()

    t = np.array(eval_callback.timesteps)
    r = np.array(eval_callback.episode_returns)

    file_path = os.path.join(
        save_dir, f"{algo_type}_halfcheetah_seed{seed}_criticprobe.npz"
    )
    save_dict = {
        "timesteps": t,
        "returns": r,
        "pi_entropies": np.array(model.pi_entropies),
        "alpha_logs": np.array(model.alpha_logs),
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
    if algo_type == "fixed_prior":
        save_dict.update({
            "kl_values": np.array(model.kl_values),
            "beta_logs": np.array(model.beta_logs),
        })

    np.savez(file_path, **save_dict)
    print(f" Saved to {file_path}")
    print(f" probe points: {len(probe_callback.probe_timesteps)} "
          f"(every {probe_every} steps)")
    return file_path


# 計画の3本（主軸2本＋必要に応じて1本）
PLAN = [
    ("pure_sac", 8),      # 這い型で確定（観察の主軸）
    ("pure_sac", 7),      # 健全（対比の主軸）
    ("fixed_prior", 8),   # 先輩版の崩壊（必要に応じて）
]


def main():
    parser = argparse.ArgumentParser(
        description="フェーズ3 ステップ2: クリティック記録つき再走（HalfCheetah-v5）"
    )
    parser.add_argument("--algo", choices=["pure_sac", "fixed_prior"],
                        default="pure_sac")
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=1_000_000)
    parser.add_argument("--probe-every", type=int, default=5000)
    parser.add_argument("--checkpoint-every", type=int, default=100000)
    parser.add_argument("--plan", action="store_true",
                        help="計画の3本（SAC 8 → SAC 7 → 先輩版 8）を順に回す")
    parser.add_argument("--smoke", action="store_true",
                        help="試し回し: 10,000 steps（記録・保存・乱数復元の確認用）")
    args = parser.parse_args()

    print("=== フェーズ3 ステップ2: クリティック記録つき再走 ===")
    print("注意: 10シード拡張ランが終わっていることを確認してから起動してください。")
    print("      ほかの学習プロセスが動いている間は起動しないでください。")

    if args.smoke:
        print("=== SMOKE MODE (10,000 steps) ===")
        run_experiment(args.algo, args.seed, total_timesteps=10_000,
                       probe_every=1000, checkpoint_every=5000)
        return

    if args.plan:
        for algo_type, seed in PLAN:
            print(f"\n==========================================")
            print(f" Running: {algo_type} | Seed: {seed} | {args.total_steps:,} steps")
            print(f"==========================================")
            run_experiment(algo_type, seed, total_timesteps=args.total_steps,
                           probe_every=args.probe_every,
                           checkpoint_every=args.checkpoint_every)
    else:
        run_experiment(args.algo, args.seed, total_timesteps=args.total_steps,
                       probe_every=args.probe_every,
                       checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
