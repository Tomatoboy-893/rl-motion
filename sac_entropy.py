# sac_entropy.py
from stable_baselines3 import SAC

class SACWithEntropyLogging(SAC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pi_entropies = []
        self.alpha_logs = []

    def train(self, gradient_steps: int, batch_size: int = 64):

        for _ in range(gradient_steps):

            replay_data = self.replay_buffer.sample(
                batch_size,
                env=self._vec_normalize_env,
            )

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(
                replay_data.observations
            )

            # ========= entropy =========
            entropy = (-log_prob).mean().item()
            self.pi_entropies.append(entropy)

            # ========= alpha =========
            if self.ent_coef_optimizer is not None:
                alpha = self.log_ent_coef.exp().item()
            else:
                alpha = float(self.ent_coef_tensor)

            self.alpha_logs.append(alpha)

            super().train(1, batch_size)
