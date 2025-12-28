###############################################
# RoAd-RL: Robust Adversarial RL Library
#
# Based on minimalRL by Seungeun Rho
###############################################

import torch
import numpy as np
from road_rl.policies.torch_policy import TorchPolicy


class PPOPolicy(TorchPolicy):
    """
    PPO (discrete) inference-only policy adapter.
    """

    def __init__(self, policy_net: torch.nn.Module, device="cpu"):
        self.pi = policy_net.to(device)
        self.pi.eval()
        self.device = device

    def act(self, obs, deterministic: bool = True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.pi(obs)
            if deterministic:
                action = torch.argmax(prob, dim=1).item()
            else:
                action = torch.distributions.Categorical(prob).sample().item()
        return action
