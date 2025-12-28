###############################################
# RoAd-RL: Robust Adversarial RL Library
#
# TorchPolicy base class
#
# This class provides a thin, explicit bridge
# between PyTorch-based policies and the
# RoAd-RL Policy interface.
###############################################

from __future__ import annotations

from abc import ABC
from typing import Any

import numpy as np
import torch

from road_rl.policies.base import Policy


class TorchPolicy(Policy, ABC):
    """
    Base class for PyTorch-based inference-only policies.

    This class handles:
    - device placement
    - torch.no_grad inference
    - observation -> torch tensor conversion

    Subclasses must implement `act`.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)

    def _obs_to_tensor(self, obs: Any) -> torch.Tensor:
        """
        Convert an observation to a torch tensor on the correct device.
        """
        if isinstance(obs, torch.Tensor):
            return obs.to(self.device)

        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs).float().to(self.device)

        # Fallback (e.g. list, tuple)
        return torch.tensor(obs, dtype=torch.float32, device=self.device)

    def eval_mode(self) -> None:
        """
        Put policy networks into eval mode.

        Subclasses should override this if they contain
        multiple networks.
        """
        pass

    def act(self, obs: Any, deterministic: bool = True):
        """
        Compute an action for a single observation.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
