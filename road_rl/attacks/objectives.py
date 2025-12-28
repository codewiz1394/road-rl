"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Attack objective functions.

This module defines scalar loss functions used by adversarial
attacks such as FGSM, PGD, and JSMA. Objectives are intentionally
kept separate from attack mechanics so they can be reused,
compared, and discussed explicitly in the paper.
"""

from __future__ import annotations

import torch

from road_rl.policies.base import DifferentiablePolicy


def negative_log_prob_loss(
    policy: DifferentiablePolicy,
    observation: torch.Tensor,
) -> torch.Tensor:
    """
    Negative log-probability of the action selected by the policy.

    This is the most common objective used in adversarial attacks
    on reinforcement learning policies. The attack attempts to
    reduce the confidence of the policy in its chosen action.
    """
    outputs = policy.forward(observation)

    if not hasattr(policy, "log_prob"):
        raise AttributeError(
            "Policy does not implement log_prob(). "
            "This objective requires action log-probabilities."
        )

    logp = policy.log_prob(outputs)
    return -logp


def value_degradation_loss(
    policy: DifferentiablePolicy,
    observation: torch.Tensor,
) -> torch.Tensor:
    """
    Value-based objective.

    The attack attempts to reduce the estimated value of the
    current state, indirectly degrading long-term performance.
    """
    if not hasattr(policy, "value"):
        raise AttributeError(
            "Policy does not implement value(). "
            "This objective requires a value estimate."
        )

    value = policy.value(observation)
    return -value


def action_margin_loss(
    policy: DifferentiablePolicy,
    observation: torch.Tensor,
) -> torch.Tensor:
    """
    Action margin loss.

    This objective reduces the margin between the best and
    second-best actions, increasing action ambiguity.
    """
    logits = policy.forward(observation)

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    top2 = torch.topk(logits, k=2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    return margin.mean()
