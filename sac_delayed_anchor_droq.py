"""
遅延アンカー版 ADR + DroQ 流クリティック (SACWithDelayedAnchorDroQ)
====================================================================

位置づけ
--------
提案手法（遅延アンカー版 ADR、sac_delayed_anchor.py）はそのままに、
クリティック（Q 関数）だけを DroQ 流に強化した併用版です。

- ADR（β・KL・アンカー）は方策（アクター）側の仕組み、
  DroQ はクリティック側の仕組みで、作用する場所が重なりません。
- 狙いは、天井と立ち上がりを決めているクリティックの質を上げて、
  遅延アンカー単体では届かなかった水準（同一条件の先輩版
  FixedPrior の IQM 約12,726）を超えられるかを確認することです。
- 出典: DroQ (Hiraoka et al., ICLR 2022) https://arxiv.org/abs/2110.02034
        REDQ (Chen et al., 2021) https://arxiv.org/abs/2101.05982

変えること（クリティック側だけ）
--------------------------------
1. Q ネットワークの各隠れ層を
   Linear → Dropout(p=0.01) → LayerNorm → ReLU の並びにする
   （DroQ 原著コードと同じ並び。ドロップアウト率 0.01 は論文
   Appendix H・Table 8 の主実験の値。ただし HalfCheetah は論文の
   実験環境に含まれておらず、環境別の再調整値は 0.0001〜0.1 と
   幅広い。0.01 は出発点であり、感度が高い前提で扱う）。
   アクター側のネットワークには一切手を入れません。
2. ターゲット Q の計算でもドロップアウトを有効にする。
   SB3 はターゲット・クリティックを常に eval モード（ドロップアウト
   無効）にするため、そのままだと DroQ 論文 Appendix C の
   アブレーション変種（-DO@TargetQ、性能が落ちると報告）になって
   しまいます。原著コードはドロップアウト率 > 0 のときターゲットを
   eval にしません。本実装では train() の中で、ターゲット Q を
   計算する前後だけ critic_target を train モードに切り替えます
   （LayerNorm には実行時統計が無いので、この切り替えで変わるのは
   ドロップアウトの on/off だけです）。
3. UTD（環境 1 ステップあたりの勾配更新回数）を 5 にする
   （gradient_steps=5。論文の 20 ではなく、計算機の壁時間に
   合わせた実務値）。

変えないこと（遅延アンカー側は従来どおり）
------------------------------------------
- アンカーの定義（Polyak 平均した旧方策、anchor_tau=0.005）、
  target_kl=0.05/次元、β の自動調整とクランプ [1e-4, 10]、
  KL 爆発への防御（アンカー側 log_std 下限 -4、項ごとの上限 10）、
  ブレーキのウォームアップ（初期設定では無効）は、すべて
  sac_delayed_anchor.py と同じです。
- gradient_steps=1 のときの学習の意味は従来版と一致します
  （違いはターゲット側ドロップアウトと追加ログだけ）。

設計上の判断（2026-11-13、実装時に確定させた扱い）
--------------------------------------------------
(a) 方策更新で使う Q は、既定では従来どおり min（2つの Q の小さい方）。
    DroQ 原著は平均を使いますが、ここでは「変える要素をクリティック
    だけに絞って、遅延アンカー単体との差の原因を切り分けられる」
    ことを優先しました。引数 policy_q_agg="mean" で原著どおりの
    平均に切り替えられます（その場合、論文での呼び名は DroQ その
    ものに近づきますが、比較の切り分けは弱くなります）。
    なおターゲット Q は、どちらの設定でも原著どおり min です。
(b) アンカーの Polyak 更新と β の更新は、勾配ステップごとではなく
    「環境ステップにつき 1 回」（train() の呼び出しごとに 1 回）に
    しました。もし勾配ステップごとに更新すると、UTD=5 のとき
    アンカーの実効速度が約 5 倍（1-(1-0.005)^5 ≈ 0.0248）になり、
    「提案手法そのもの」が変わってしまうためです。β の調整も、
    本実験（UTD=1）と同じ速さに揃えます。クリティックの
    ターゲットネットの Polyak 更新は、従来どおり勾配ステップ
    ごとです（高 UTD の設計では通常の扱い）。

追加のログ
----------
従来の診断（kl_values, kl_raw_values, kl_mean_terms, kl_std_terms,
beta_logs, brake_factors, pi_entropies, alpha_logs）に加えて、
- critic_losses : クリティック損失（勾配ステップごと）
- q_means       : バッチ平均の Q 値（(Q1+Q2)/2、勾配ステップごと）
を記録します。Q の発散・NaN の監視用です。
※ beta_logs は環境ステップごとの記録、他は勾配ステップごとの
   記録です（UTD=5 では点数が 5 倍違うので、解釈時に注意）。

注意（SB3 のバージョン依存）
----------------------------
方策クラスの差し替え（make_critic のオーバーライド）と
ContinuousCritic の q_networks の作り直しは、SB3 master の
実装に合わせて書いてあります。keisanB の SB3 のバージョンが
違うと、メソッドの署名や内部構造が異なる可能性があります。
まずスモークテスト（10k steps）で、モデルの構築と学習が
通ることを確認してください。
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from sac_delayed_anchor import SACWithDelayedAnchorPrior


class DroQContinuousCritic(ContinuousCritic):
    """SB3 の ContinuousCritic を継承し、q_networks だけを作り直す。

    親が作る各 q_net は [Linear, ReLU, Linear, ReLU, Linear] なので、
    子要素を順にたどり、最終層以外の Linear の直後に
    Dropout と LayerNorm を挿入し、
    [Linear, Dropout, LayerNorm, ReLU, Linear, Dropout, LayerNorm,
     ReLU, Linear] に組み替えます（DroQ 原著コードと同じ並び）。
    forward / q1_forward は親のものをそのまま使います。
    """

    def __init__(self, *args, dropout_rate: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        new_q_nets = []
        for idx, q_net in enumerate(self.q_networks):
            layers = list(q_net.children())
            rebuilt = []
            for j, layer in enumerate(layers):
                rebuilt.append(layer)
                if isinstance(layer, nn.Linear) and j < len(layers) - 1:
                    rebuilt.append(nn.Dropout(p=dropout_rate))
                    rebuilt.append(nn.LayerNorm(layer.out_features))
            new_net = nn.Sequential(*rebuilt)
            # 親が登録した qf{idx} を差し替え、q_networks も差し替える
            self.add_module(f"qf{idx}", new_net)
            new_q_nets.append(new_net)
        self.q_networks = new_q_nets


class DroQSACPolicy(SACPolicy):
    """make_critic() だけを差し替える方策クラス。

    policy_kwargs にクリティック・クラスの差し替え口が無いため、
    方策クラスごと差し替えます。アクターの作り方は親のままです。
    ドロップアウト率はクラス属性で持ちます（変えたい場合は
    モデル生成前に DroQSACPolicy.dropout_rate を書き換える）。
    """

    dropout_rate = 0.01

    def make_critic(self, features_extractor=None):
        critic_kwargs = self._update_features_extractor(
            self.critic_kwargs, features_extractor
        )
        return DroQContinuousCritic(
            dropout_rate=self.dropout_rate, **critic_kwargs
        ).to(self.device)


class SACWithDelayedAnchorDroQ(SACWithDelayedAnchorPrior):
    """遅延アンカー版 ADR に DroQ 流クリティックを併用した版。

    Parameters
    ----------
    policy_q_agg : str
        方策更新で使う Q のまとめ方。"min"（既定、従来の SAC と同じ）
        または "mean"（DroQ 原著どおり）。ターゲット Q は常に min。
    その他の引数は SACWithDelayedAnchorPrior と同じです
    （policy には DroQSACPolicy を渡してください）。
    """

    def __init__(self, *args, policy_q_agg: str = "min", **kwargs):
        if policy_q_agg not in ("min", "mean"):
            raise ValueError(
                f"policy_q_agg は 'min' か 'mean' です: {policy_q_agg}"
            )
        super().__init__(*args, **kwargs)
        self.policy_q_agg = policy_q_agg
        # 追加ログ（Q の発散・NaN の監視用）
        self.critic_losses = []  # クリティック損失（勾配ステップごと）
        self.q_means = []        # バッチ平均 Q 値（勾配ステップごと）

    def train(self, gradient_steps: int, batch_size: int = 64):
        # sac_delayed_anchor.py の train() を土台に、DroQ 併用に必要な
        # 変更だけを加えた複製です。変更点には ★DroQ の印を付けます。
        self.policy.set_training_mode(True)

        last_kl_mean = None

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

            # 1. クリティック更新
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.unsqueeze(-1)
                # ★DroQ: ターゲット Q でもドロップアウトを効かせる。
                #   SB3 の既定では critic_target は eval モードで
                #   ドロップアウトが切れてしまうため、計算の前後だけ
                #   train モードにする（原著コードの扱いに合わせる）。
                self.critic_target.train()
                next_q = torch.min(*self.critic_target(next_obs, next_actions))
                self.critic_target.eval()
                alpha = torch.exp(self.log_ent_coef.detach())
                target_q = rewards + (1 - dones) * self.gamma * (
                    next_q - alpha * next_log_prob
                )

            current_q1, current_q2 = self.critic(obs, actions)
            critic_loss = (
                torch.nn.functional.mse_loss(current_q1, target_q)
                + torch.nn.functional.mse_loss(current_q2, target_q)
            )

            # ★DroQ: Q の発散・NaN 監視用の記録
            self.critic_losses.append(float(critic_loss.detach().cpu()))
            q_mean = 0.5 * (current_q1 + current_q2)
            self.q_means.append(float(q_mean.mean().detach().cpu()))

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # 2. 今のポリシーとアンカーの分布パラメータ取得（従来どおり）
            pi_mean, pi_log_std, _ = self.actor.get_action_dist_params(obs)
            pi_log_std = torch.clamp(pi_log_std, -20.0, 2.0)

            with torch.no_grad():
                anchor_mean, anchor_log_std, _ = (
                    self.anchor_actor.get_action_dist_params(obs)
                )
                anchor_log_std = torch.clamp(
                    anchor_log_std, self.anchor_log_std_min, 2.0
                )

            pi_dist = Normal(pi_mean, torch.exp(pi_log_std))
            entropy = pi_dist.entropy().sum(dim=-1).mean()
            self.pi_entropies.append(float(entropy.detach().cpu()))

            # 3. KL(π || アンカー) の計算と記録（従来どおり）
            kl, kl_mean_part, kl_std_part, kl_raw = self.kl_diag_gaussians(
                pi_mean, pi_log_std, anchor_mean, anchor_log_std,
                term_cap=self.kl_term_cap,
            )
            kl_mean = kl.mean()
            last_kl_mean = kl_mean
            self.kl_values.append(float(kl_mean.detach().cpu()))
            self.kl_raw_values.append(float(kl_raw.mean().detach().cpu()))
            self.kl_mean_terms.append(float(kl_mean_part.mean().detach().cpu()))
            self.kl_std_terms.append(float(kl_std_part.mean().detach().cpu()))

            # 4. アクター更新（SAC の損失 + β × KL、従来どおり）
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.unsqueeze(-1)
            # ★DroQ: 方策 Q のまとめ方は policy_q_agg で選ぶ
            #   （既定 "min" は従来どおり、"mean" は DroQ 原著どおり）
            q1_pi, q2_pi = self.critic(obs, actions_pi)
            if self.policy_q_agg == "mean":
                q_pi = 0.5 * (q1_pi + q2_pi)
            else:
                q_pi = torch.min(q1_pi, q2_pi)

            beta = torch.exp(self.log_beta).clamp(1e-12, 1e6)
            alpha = torch.exp(self.log_ent_coef.detach())

            brake_factor = self._brake_factor()
            self.brake_factors.append(brake_factor)
            beta_eff = beta * brake_factor

            actor_loss = (
                alpha * log_prob - q_pi + beta_eff * kl.unsqueeze(-1)
            ).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

            # 5. エントロピー係数 α の更新（従来どおり）
            ent_coef_loss = -(
                self.log_ent_coef * (log_prob + self.target_entropy).detach()
            ).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            alpha = torch.exp(self.log_ent_coef.detach())
            self.alpha_logs.append(float(alpha.cpu()))

            # 6. クリティックのターゲットネット更新（勾配ステップごと）
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau,
                )

            self._n_updates += 1

        # ★DroQ: ここから先は「環境ステップにつき 1 回」
        #   （train() の呼び出しごとに 1 回）だけ行います。
        #   勾配ステップごとに回すと、UTD を上げた分だけアンカーの
        #   追従と β の調整が速くなり、提案手法そのものが変わって
        #   しまうためです（設計上の判断 (b)）。
        if last_kl_mean is not None:
            # 7. β の自動調整（最後の勾配ステップの KL を使う）
            beta_loss = -self.log_beta * (
                last_kl_mean.detach() - self.target_kl
            )
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()

            with torch.no_grad():
                self.log_beta.clamp_(self.log_beta_min, self.log_beta_max)

            beta_now = torch.exp(self.log_beta.detach())
            self.beta_logs.append(float(beta_now.cpu()))

            # 8. アンカーアクターの追従（係数は anchor_tau、従来どおり）
            polyak_update(
                self.actor.parameters(),
                self.anchor_actor.parameters(),
                self.anchor_tau,
            )
