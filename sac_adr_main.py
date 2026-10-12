import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.utils import clip_grad_norm_

from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


# ===============
# ADRの実装クラス (固定事前分布)
# ===============
class SACWithFixedPrior(SAC):

    def __init__(
        self,
        *args,
        beta_kl,
        beta_lr,
        target_kl,
        prior_std,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        act_dim = self.action_space.shape[0]

        self.prior_mean = torch.zeros(act_dim, device=self.device)
        self.prior_log_std = torch.log(
            torch.ones(act_dim, device=self.device) * prior_std
        )

        init_log_beta = math.log(max(beta_kl, 1e-12))
        self.log_beta = nn.Parameter(
            torch.tensor(init_log_beta, device=self.device)
        )

        self.beta_optimizer = torch.optim.Adam([self.log_beta], lr=beta_lr)
        self.target_kl = act_dim * target_kl

        self.kl_values = []
        self.beta_logs = []
        self.pi_entropies = []
        self.alpha_logs = []

    @staticmethod
    def kl_diag_gaussians(mu1, log_std1, mu2, log_std2):
        var1 = torch.exp(2 * log_std1)
        var2 = torch.exp(2 * log_std2)
        kl = (
            log_std2 - log_std1
            + (var1 + (mu1 - mu2) ** 2) / (2.0 * var2)
            - 0.5
        )
        return kl.sum(dim=-1)

    def train(self, gradient_steps: int, batch_size: int = 64):
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

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

            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.unsqueeze(-1)

            q_pi = torch.min(*self.critic(obs, actions_pi))

            beta = torch.exp(self.log_beta)
            alpha = torch.exp(self.log_ent_coef.detach())

            pi_mean, pi_log_std, _ = self.actor.get_action_dist_params(obs)
            pi_log_std = torch.clamp(pi_log_std, -20.0, 2.0)
            pi_dist = Normal(pi_mean, torch.exp(pi_log_std))

            entropy = pi_dist.entropy().sum(dim=-1).mean()
            self.pi_entropies.append(float(entropy.detach().cpu()))

            prior_mean = self.prior_mean.unsqueeze(0).expand_as(pi_mean)
            prior_log_std = self.prior_log_std.unsqueeze(0).expand_as(pi_log_std)

            kl = self.kl_diag_gaussians(pi_mean, pi_log_std, prior_mean, prior_log_std)
            kl_mean = kl.mean()
            self.kl_values.append(float(kl_mean.detach().cpu()))

            actor_loss = (alpha * log_prob - q_pi + beta * kl.unsqueeze(-1)).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            alpha = torch.exp(self.log_ent_coef.detach())
            self.alpha_logs.append(float(alpha.cpu()))

            beta_loss = -self.log_beta * (kl_mean.detach() - self.target_kl)
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()

            beta_clamped = torch.exp(self.log_beta).clamp(1e-12, 1e6)
            self.beta_logs.append(float(beta_clamped.detach().cpu()))

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            self._n_updates += 1


# ===============
# 移動平均プライオ版の拡張クラス
# ===============
class SACWithMovingAveragePrior(SACWithFixedPrior):
    def __init__(
        self,
        *args,
        tau: float = 0.005,
        use_value_filter: bool = True,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.tau = tau
        self.use_value_filter = use_value_filter
        
        act_dim = self.action_space.shape[0]
        self.dynamic_prior_mean = torch.zeros(act_dim, device=self.device)
        self.prior_mean_logs = []

    def train(self, gradient_steps: int, batch_size: int = 64):
        self.policy.set_training_mode(True)

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size)
            obs = replay_data.observations
            actions = replay_data.actions
            next_obs = replay_data.next_observations
            rewards = replay_data.rewards
            dones = replay_data.dones

            with torch.no_grad():
                actions_pi, _ = self.actor.action_log_prob(obs)

                if self.use_value_filter:
                    current_q1, current_q2 = self.critic(obs, actions_pi)
                    current_q = torch.min(current_q1, current_q2)
                    q_median = current_q.median()
                    mask = current_q >= q_median
                    
                    if mask.sum() > 0:
                        filtered_actions = actions_pi[mask]
                        batch_mean = filtered_actions.mean(dim=0)
                    else:
                        batch_mean = actions_pi.mean(dim=0)
                else:
                    batch_mean = actions_pi.mean(dim=0)

                self.dynamic_prior_mean = (
                    (1.0 - self.tau) * self.dynamic_prior_mean 
                    + self.tau * batch_mean.detach()
                )
                self.prior_mean_logs.append(self.dynamic_prior_mean.cpu().numpy().copy())

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

            actions_pi, log_prob = self.actor.action_log_prob(obs)
            log_prob = log_prob.unsqueeze(-1)

            q_pi = torch.min(*self.critic(obs, actions_pi))

            beta = torch.exp(self.log_beta)
            alpha = torch.exp(self.log_ent_coef.detach())

            pi_mean, pi_log_std, _ = self.actor.get_action_dist_params(obs)
            pi_log_std = torch.clamp(pi_log_std, -20.0, 2.0)
            pi_dist = Normal(pi_mean, torch.exp(pi_log_std))

            entropy = pi_dist.entropy().sum(dim=-1).mean()
            self.pi_entropies.append(float(entropy.detach().cpu()))

            prior_mean = self.dynamic_prior_mean.unsqueeze(0).expand_as(pi_mean)
            prior_log_std = self.prior_log_std.unsqueeze(0).expand_as(pi_log_std)

            kl = self.kl_diag_gaussians(pi_mean, pi_log_std, prior_mean, prior_log_std)
            kl_mean = kl.mean()
            self.kl_values.append(float(kl_mean.detach().cpu()))

            actor_loss = (alpha * log_prob - q_pi + beta * kl.unsqueeze(-1)).mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor.optimizer.step()

            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()

            alpha = torch.exp(self.log_ent_coef.detach())
            self.alpha_logs.append(float(alpha.cpu()))

            beta_loss = -self.log_beta * (kl_mean.detach() - self.target_kl)
            self.beta_optimizer.zero_grad()
            beta_loss.backward()
            self.beta_optimizer.step()

            beta_clamped = torch.exp(self.log_beta).clamp(1e-12, 1e6)
            self.beta_logs.append(float(beta_clamped.detach().cpu()))

            if self._n_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

            self._n_updates += 1
