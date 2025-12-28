###############################################
# RoAd-RL: Robust Adversarial RL Library
#
# Based on minimalRL by Seungeun Rho
# https://github.com/seungeunrho/minimalRL
###############################################

import torch
import numpy as np

from road_rl.policies.torch_policy import TorchPolicy


class DQNPolicy(TorchPolicy):
    """
    DQN inference-only policy adapter.
    """

    def __init__(self, q_network: torch.nn.Module, device="cpu"):
        self.q_net = q_network.to(device)
        self.q_net.eval()
        self.device = device

    def act(self, obs, deterministic: bool = True):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs)
            action = torch.argmax(q_values, dim=1).item()
        return action
