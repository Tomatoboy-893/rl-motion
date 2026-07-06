import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


# ===============
# ADRの実装クラス
# ===============
class SACWithFixedPrior(SAC):

    def __init__(
        self,
        *args,
        beta_kl, # KL正則化にかけるβの初期値
        beta_lr, # βの学習率
        target_kl, # 目標KL値(αの目標値と同様→KLは次元ごとなので1.0*action_dimでスケーリング)
        prior_std, # 幅広の正規分布の分散
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # action次元の取得(HalfCheetah-v5なら6)→KLでは次元ごとの足し合わせで必要
        act_dim = self.action_space.shape[0]

        # 固定 prior の定義→状態非依存、学習しない、σが大きい分布→探索を事前分布に委ねる設計
        self.prior_mean = torch.zeros(act_dim, device=self.device)
        self.prior_log_std = torch.log(
            torch.ones(act_dim, device=self.device) * prior_std
        )

        # βを学習可能にする→なぜlog_β？→αと同じ理論でβ>0を保証するため(勾配が安定)
        init_log_beta = math.log(max(beta_kl, 1e-12))
        self.log_beta = nn.Parameter(
            torch.tensor(init_log_beta, device=self.device)
        )

        # β用のoptimizer
        self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr)

        # 目標KL値の定義→αのエントロピー=-dim(A)に対応して、ADRでは次元を考慮してスケール
        self.target_kl = act_dim * target_kl

        # ログ保存
        self.kl_values = []
        self.beta_logs = []
        self.pi_entropies = []
        self.alpha_logs = []

    # ================================================================
    # KL(π || prior) 計算→ADRのコア部分(πがpriorからどれだけ離れているか)
    # ================================================================
    @staticmethod
    # mu1,log_std1とmu2,log_std2のKLをバッチ×アクション次元ごとに計算
    def kl_diag_gaussians(mu1, log_std1, mu2, log_std2):
        # 分散の計算→sacのactor出力と完全一致
        var1 = torch.exp(2 * log_std1)
        var2 = torch.exp(2 * log_std2)
        # KLの中身(次元ごと)
        kl = (
            log_std2 - log_std1
            + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2)
            - 0.5
        )
        # 次元方向の総和
        return kl.sum(dim=-1)

    # ====================================================
    # Training (fully overridden)
    # ====================================================
    def train(self, gradient_steps: int, batch_size: int = 64):

        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):

            # ============================================
            # リプレイバッファからのサンプリング(SACそのまま)
            # ============================================
            replay_data = self.replay_buffer.sample(batch_size)
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

            # =============================================
            # Criticの更新(SACそのまま)→ADRはCriticに触れない
            # =============================================
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
                next_log_prob = next_log_prob.unsqueeze(-1)

                next_q = torch.min(
                    *self.critic_target(next_obs, next_actions)
                )

                alpha = torch.exp(self.log_ent_coef.detach())

                target_q = rewards + (1 - dones) * self.gamma * (
                    next_q - alpha * next_log_prob
                )

            current_q1, current_q2 = self.critic(obs, actions)
            critic_loss = (
                torch.nn.functional.mse_loss(current_q1, target_q)
                + torch.nn.functional.mse_loss(current_q2, target_q)
            )

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # =====================================
            # Actorの更新→ここにADRアルゴリズムを実装
            # =====================================
            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.unsqueeze(-1)

            q_pi = torch.min(*self.critic(obs, actions_pi))

            beta = torch.exp(self.log_beta)
            alpha = torch.exp(self.log_ent_coef.detach())

            # πの分布を明示的に作成→KL計算のため(mean,stdが必要)
            pi_mean, pi_log_std, _ = self.actor.get_action_dist_params(obs)
            pi_log_std = torch.clamp(pi_log_std, -20.0, 2.0)
            pi_dist = Normal(pi_mean, torch.exp(pi_log_std))

            # entropy log
            entropy = pi_dist.entropy().sum(dim=-1).mean()
            self.pi_entropies.append(float(entropy.detach().cpu()))

            # priorの展開→バッチサイズに合わせて拡張、状態非依存
            prior_mean = self.prior_mean.unsqueeze(0).expand_as(pi_mean)
            prior_log_std = self.prior_log_std.unsqueeze(0).expand_as(pi_log_std)

            # KL(π || prior)の計算→ADRの正則化項
            kl = self.kl_diag_gaussians(
                pi_mean, pi_log_std, prior_mean, prior_log_std
            )
            kl_mean = kl.mean()
            self.kl_values.append(float(kl_mean.detach().cpu()))

            beta = torch.exp(self.log_beta)

            # ADR の定義そのもの(Actorの損失にKL正則化項を追加)
            actor_loss = (
                alpha * log_prob
                - q_pi
                + beta * kl.unsqueeze(-1)
            ).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            # 勾配の制御
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

            # =======
            # αの更新
            # ======
            ent_coef_loss = -(
                self.log_ent_coef * (log_prob + self.target_entropy).detach()
            ).mean()

            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            alpha = torch.exp(self.log_ent_coef.detach())
            self.alpha_logs.append(float(alpha.cpu()))

            # ===================================================================================
            # βの更新→KLが大きいと(離れていると)βを大きくする、KLが小さいと(近づいていると)βを小さくする
            # ===================================================================================
            beta_loss = -self.log_beta * (kl_mean.detach() - self.target_kl)

            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()

            beta_clamped = torch.exp(self.log_beta).clamp(1e-12, 1e6)
            self.beta_logs.append(float(beta_clamped.detach().cpu()))

            # =================================
            # Target network update(SACそのまま→ADR非依存)
            # =================================
            if self._n_updates % self.target_update_interval == 0:
                polyak_update(
                    self.critic.parameters(),
                    self.critic_target.parameters(),
                    self.tau
                )

            self._n_updates += 1
