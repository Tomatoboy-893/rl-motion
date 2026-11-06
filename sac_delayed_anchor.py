"""
遅延アンカー版 ADR (SACWithDelayedAnchorPrior)
================================================

考え方（ざっくり）
------------------
事前分布（＝お手本）を「今のポリシーそのもの」ではなく、
「少し前のポリシー（ゆっくり追いかけてくるコピー）」にする版です。

- 従来の PolicyMeanPrior 版は、お手本の平均が今のポリシーの平均と
  常に一致していたため、KL の平均項（平均同士の距離）が常に 0 になり、
  「引っ張る力」が消えてしまっていました。
- この版では、お手本（アンカー）は少し遅れて追いかけてくるので、
  ポリシーが動いた分だけ距離（KL）が生まれ、引っ張る力が残ります。
  ただしアンカーもゆっくり追いつくため、止められるのは
  「急な変化」だけで、「ゆっくりした変化」は通ります。
  狙いはリターンの底上げではなく、学習の安定化
  （急な更新で歩行が壊れて崩れるのを防ぐこと）です。

仕組み
------
- アクターのコピーを1つ持ち（アンカーアクター）、
  学習ステップごとに Polyak 平均でゆっくり現在に追いつかせます。
  ※ 追従係数 anchor_tau は、クリティックのターゲットネット用である
    self.tau とは必ず分離します（過去に共有して上書きされる問題があったため）。
- 事前分布は、状態ごとにアンカーアクターが出す分布
  ρ(·|s) = N( μ_anchor(s), σ_anchor(s)^2 ) とします。
- KL(π || ρ) は tanh 前の潜在ガウス同士で解析的に計算します
  （既存コードと同じ流儀です）。
- KL の重み β は、KL の実測値が target_kl に近づくよう
  双対目的で自動調整します（先輩・従来版と同じ仕掛け）。

注意（前回の Humanoid 実験の反省から入れた対策）
------------------------------------------------
1. target_kl の初期値は従来の 1.0/次元 よりずっと小さくしています。
   遅延アンカー同士の KL はもともと小さい量なので、目標が大きすぎると
   「KL が目標に届かない → β が下がり続ける → β≈0 で装置が消える」
   という前回の失敗を繰り返します。
   まずは kl_values のログを見て「自然な KL の大きさ」を確認し、
   その近辺に target_kl を置くのが前提です（初期値は目安）。
2. β には下限 beta_min を設け、log β を範囲内にクランプします。
   β が完全に 0 へ落ちて二度と戻らない事態を防ぐためです。
3. ログの beta_logs / alpha_logs に入っているのは
   「log ではなく β・α そのものの値」です（従来の図で
   タイトルと中身がずれていた件の混同防止のため、ここに明記します）。
4. KL は「平均項（行動の平均のズレ）」と「σ項（ばらつきのズレ）」に
   分けて別々に記録します。どちらの力が効いているかを
   後から切り分けられるようにするためです。
5. 【試し回しで実際に起きた KL 爆発への対策】
   100k steps の試し回しで、開始直後に KL が約 9×10^16 まで
   爆発しました。原因は、一部の状態・次元でアンカー側の σ が
   ほぼ 0 になり、平均項 (Δμ)^2 / (2 σ_anchor^2) の「分母」が
   ほぼ 0 の割り算になったことです。対策は2段構えです。
   (a) アンカー側 log_std の下限を -20 から -4 へ引き上げる
       （σ の下限が約 0.018 になり、分母が潰れなくなります）。
   (b) 平均項・σ項に「次元ごと」の上限 kl_term_cap を設ける
       （1つの次元だけが全体を爆発させることを防ぎます）。
   上限をかける前の KL は kl_raw_values に別途記録するので、
   「上限がどれくらい頻繁に効いているか」は後から確認できます。
"""

import copy
import math

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class SACWithDelayedAnchorPrior(SAC):
    """事前分布を「少し前のポリシー（アンカーアクター）」にする SAC + ADR

    Parameters
    ----------
    beta_kl : float
        β の初期値。
    beta_lr : float
        β の自動調整（log β の Adam）の学習率。
    target_kl : float
        KL の目標値（**1次元あたり**）。内部で行動次元数を掛けて使います。
        遅延アンカー同士の KL は小さいので、従来版の 1.0 より
        ずっと小さい値が前提です。kl_values の実測を見て調整してください。
    anchor_tau : float
        アンカーアクターの追従係数（Polyak 平均）。
        小さいほどアンカーの遅れが大きく（＝ブレーキが強く長く）なります。
        self.tau（クリティックのターゲット用）とは独立です。
    beta_min, beta_max : float
        β の下限・上限。log β をこの範囲に対応する値へクランプし、
        β が 0 に落ち切って装置が消えるのを防ぎます。
    anchor_log_std_min : float
        アンカー側 log_std の下限。π 側の慣習（-20）のままだと
        KL の分母 σ_anchor^2 がほぼ 0 になり、平均項が爆発します
        （試し回しで実測）。-4（σ≈0.018）まで引き上げます。
    kl_term_cap : float
        平均項・σ項それぞれに対する「1次元あたり」の上限。
        1つの次元の異常値が KL 全体を爆発させるのを防ぎます。
    """

    def __init__(
        self,
        *args,
        beta_kl=0.01,
        beta_lr=1e-3,
        target_kl=0.05,
        anchor_tau=0.005,
        beta_min=1e-4,
        beta_max=10.0,
        anchor_log_std_min=-4.0,
        kl_term_cap=10.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        act_dim = self.action_space.shape[0]

        # --- アンカーアクター（少し前の自分）を作る ---
        # 現在のアクターを丸ごとコピーし、勾配計算の対象から外します。
        # 以後、学習ステップごとに Polyak 平均で少しずつ現在に追いつかせます。
        self.anchor_actor = copy.deepcopy(self.actor)
        self.anchor_actor.set_training_mode(False)
        for param in self.anchor_actor.parameters():
            param.requires_grad_(False)
        self.anchor_tau = anchor_tau

        # --- β（KL の重み）の自動調整まわり ---
        # β は log β として保持し、exp することで β > 0 を保証します。
        init_log_beta = math.log(max(beta_kl, 1e-12))
        self.log_beta = nn.Parameter(
            torch.tensor(init_log_beta, device=self.device)
        )
        self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr)

        # 目標 KL は「1次元あたり」で受け取り、内部では全次元の合計に直します
        self.target_kl = act_dim * target_kl

        # β の暴走・消失を防ぐための範囲（log 空間で保持）
        self.log_beta_min = math.log(beta_min)
        self.log_beta_max = math.log(beta_max)

        # --- KL 爆発への2つの防御（試し回しの反省） ---
        self.anchor_log_std_min = anchor_log_std_min  # アンカーσの下限
        self.kl_term_cap = kl_term_cap                # 項ごとの上限（次元ごと）

        # --- ログ置き場 ---
        # ※ beta_logs / alpha_logs の中身は log ではなく「値そのもの」です
        self.kl_values = []       # KL（全次元合計、上限適用後＝実際に使う値）
        self.kl_raw_values = []   # KL（上限適用前。σの下限だけは適用済み）
        self.kl_mean_terms = []   # KL のうち「平均項」だけの合計（上限適用後）
        self.kl_std_terms = []    # KL のうち「σ項」だけの合計（上限適用後）
        self.beta_logs = []       # β そのものの値
        self.pi_entropies = []    # 方策エントロピー（tanh 前ガウスの合計）
        self.alpha_logs = []      # α そのものの値

    @staticmethod
    def kl_diag_gaussians(mu1, log_std1, mu2, log_std2, term_cap=None):
        """2つの対角ガウス分布の KL(π1 || π2) を計算し、内訳も返します。

        戻り値は (KL合計, 平均項の合計, σ項の合計, 上限適用前のKL合計) で、
        いずれも行動次元方向に合計済み・サンプルごとの形 [batch] です。

        - 平均項: (μ1 - μ2)^2 / (2 σ2^2)
          「行動の平均がアンカーからどれだけ動いたか」を表します。
          今のポリシーをお手本にした従来版では、ここが常に 0 でした。
        - σ項: log(σ2/σ1) + σ1^2 / (2 σ2^2) - 1/2
          「ばらつきの大きさのズレ」を表します。

        term_cap が指定された場合、平均項・σ項のそれぞれに
        「次元ごと」の上限をかけてから合計します。1つの次元で
        σ2 が極小になるなどして項が異常に大きくなっても、
        KL 全体が爆発しないようにするための防御です。
        （上限適用前の値も4番目に返すので、防御が効いたかどうかは
        ログで追跡できます。）
        """
        var1 = torch.exp(2 * log_std1)
        var2 = torch.exp(2 * log_std2)

        mean_term = (mu1 - mu2) ** 2 / (2.0 * var2)
        std_term = log_std2 - log_std1 + var1 / (2.0 * var2) - 0.5

        # 上限をかける前の合計（監視用）
        kl_raw = (mean_term + std_term).sum(dim=-1)

        if term_cap is not None:
            mean_term = torch.clamp(mean_term, max=term_cap)
            std_term = torch.clamp(std_term, max=term_cap)

        kl = mean_term + std_term
        return kl.sum(dim=-1), mean_term.sum(dim=-1), std_term.sum(dim=-1), kl_raw

    def train(self, gradient_steps: int, batch_size: int = 64):
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

            # 1. クリティック更新（通常の SAC と同じ）
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.unsqueeze(-1)
                next_q = torch.min(*self.critic_target(next_obs, next_actions))
                alpha = torch.exp(self.log_ent_coef.detach())
                target_q = rewards + (1 - dones) * self.gamma * (next_q - alpha * next_log_prob)

            current_q1, current_q2 = self.critic(obs, actions)
            critic_loss = (
                torch.nn.functional.mse_loss(current_q1, target_q)
                + torch.nn.functional.mse_loss(current_q2, target_q)
            )

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # 2. 今のポリシーとアンカー（少し前のポリシー）の分布パラメータ取得
            pi_mean, pi_log_std, _ = self.actor.get_action_dist_params(obs)
            pi_log_std = torch.clamp(pi_log_std, -20.0, 2.0)

            with torch.no_grad():
                # アンカー側は「お手本」なので勾配は流しません
                anchor_mean, anchor_log_std, _ = (
                    self.anchor_actor.get_action_dist_params(obs)
                )
                # ★ 下限は π 側（-20）と別にします。アンカーの σ は
                #   KL の「分母」に入るので、ほぼ 0 になると平均項が
                #   爆発します（試し回しで約 9×10^16 を実測）。
                anchor_log_std = torch.clamp(
                    anchor_log_std, self.anchor_log_std_min, 2.0
                )

            # エントロピーの記録（従来コードと同じ定義）
            pi_dist = Normal(pi_mean, torch.exp(pi_log_std))
            entropy = pi_dist.entropy().sum(dim=-1).mean()
            self.pi_entropies.append(float(entropy.detach().cpu()))

            # 3. KL(π || アンカー) の計算と記録（平均項／σ項に分けて残す）
            #    損失と β の調整に使うのは「上限適用後」の KL です。
            #    上限適用前の KL も別に記録し、防御が効いた頻度を追います。
            kl, kl_mean_part, kl_std_part, kl_raw = self.kl_diag_gaussians(
                pi_mean, pi_log_std, anchor_mean, anchor_log_std,
                term_cap=self.kl_term_cap,
            )
            kl_mean = kl.mean()
            self.kl_values.append(float(kl_mean.detach().cpu()))
            self.kl_raw_values.append(float(kl_raw.mean().detach().cpu()))
            self.kl_mean_terms.append(float(kl_mean_part.mean().detach().cpu()))
            self.kl_std_terms.append(float(kl_std_part.mean().detach().cpu()))

            # 4. アクター更新（SAC の損失 + β × KL がこの手法の追加分）
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.unsqueeze(-1)
            q_pi = torch.min(*self.critic(obs, actions_pi))

            beta = torch.exp(self.log_beta).clamp(1e-12, 1e6)
            alpha = torch.exp(self.log_ent_coef.detach())

            actor_loss = (alpha * log_prob - q_pi + beta * kl.unsqueeze(-1)).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

            # 5. エントロピー係数 α の更新（通常の SAC と同じ）
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            alpha = torch.exp(self.log_ent_coef.detach())
            self.alpha_logs.append(float(alpha.cpu()))

            # 6. β の自動調整
            # KL が目標より大きければ β を上げ、 小さければ β を下げます。
            beta_loss = -self.log_beta * (kl_mean.detach() - self.target_kl)
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()

            # β が 0 に落ち切ったり発散したりしないよう、範囲内に収めます
            with torch.no_grad():
                self.log_beta.clamp_(self.log_beta_min, self.log_beta_max)

            beta_now = torch.exp(self.log_beta.detach())
            self.beta_logs.append(float(beta_now.cpu()))

            # 7. ターゲットネットの更新
            # (a) クリティックのターゲット（通常の SAC と同じ、係数は self.tau）
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            # (b) アンカーアクターの追従（係数は anchor_tau で、(a) とは別物）
            #     アンカーは「ゆっくり追いつく」ことが役割なので毎ステップ更新します
            polyak_update(self.actor.parameters(), self.anchor_actor.parameters(), self.anchor_tau)

            self._n_updates += 1
