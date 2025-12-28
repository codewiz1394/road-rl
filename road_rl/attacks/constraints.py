"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Attack constraint and projection utilities.

This module defines reusable functions that enforce norm-based
constraints on adversarial perturbations. These functions are
used by gradient-based attacks such as FGSM and PGD.
"""

from __future__ import annotations

import torch


def project_linf(
    adv: torch.Tensor,
    clean: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Project an adversarial example onto an L-infinity ball
    around the clean observation.
    """
    delta = adv - clean
    delta = torch.clamp(delta, min=-epsilon, max=epsilon)
    return clean + delta


def project_l2(
    adv: torch.Tensor,
    clean: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """
    Project an adversarial example onto an L2 ball
    around the clean observation.
    """
    delta = adv - clean
    norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
    norm = torch.clamp(norm, min=1e-12)

    factor = torch.min(
        torch.ones_like(norm),
        epsilon / norm,
    )

    delta = delta * factor.view(-1, *([1] * (delta.dim() - 1)))
    return clean + delta


def clip_to_bounds(
    obs: torch.Tensor,
    low: torch.Tensor | float,
    high: torch.Tensor | float,
) -> torch.Tensor:
    """
    Clip observations to valid environment bounds.
    """
    return torch.max(torch.min(obs, high), low)
