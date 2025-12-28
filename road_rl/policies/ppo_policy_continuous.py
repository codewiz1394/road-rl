###############################################
# RoAd-RL: Robust Adversarial RL Library
#
# Based on minimalRL by Seungeun Rho
###############################################

import torch
import numpy as np
from road_rl.policies.torch_policy import TorchPolicy


class PPOContinuousPolicy(TorchPolicy):
    """
    PPO (continuous) inference-only policy adapter.
    """

    def __init__(self, mu_net: torch.nn.Module, log_std: torch.Tensor, device="cpu"):
        self.mu = mu_net.to(device)
        self.log_std = log_std.to(device)
        self.mu.eval()
        self.device = device

    def act(self, obs, deterministic: bool = True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu = self.mu(obs)
            std = torch.exp(self.log_std)
            if deterministic:
                action = mu
            else:
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()
        return action.squeeze(0).cpu().numpy()
