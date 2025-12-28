"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

FGSM (Fast Gradient Sign Method) attack implementation.

FGSM is the simplest gradient-based adversarial attack:
    x_adv = x + epsilon * sign(grad_x loss(x))

This module focuses on observation-space attacks. Action-space
attacks will be added later as a separate interface.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Literal

import torch

from road_rl.attacks.base import GradientAttack
from road_rl.attacks.constraints import project_linf, project_l2, clip_to_bounds
from road_rl.core.context import StepContext
from road_rl.policies.base import Policy


NormType = Literal["linf", "l2"]


class FGSMAttack(GradientAttack):
    """
    FGSM observation attack.

    Parameters
    ----------
    norm:
        Norm constraint used for the perturbation. Common choice is "linf".

    clip_bounds:
        If True, clip the attacked observation to [low, high] bounds
        when these bounds are provided at runtime.

    objective_fn:
        Optional custom objective. If not provided, the policy's own
        loss(observation) is used.
    """

    def __init__(
        self,
        norm: NormType = "linf",
        clip_bounds: bool = True,
        objective_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        if norm not in ("linf", "l2"):
            raise ValueError(f"Unsupported norm '{norm}'. Use 'linf' or 'l2'.")
        self.norm = norm
        self.clip_bounds = bool(clip_bounds)
        self.objective_fn = objective_fn

    def apply(
        self,
        observation: Any,
        policy: Policy,
        context: StepContext,
        *,
        low: Optional[torch.Tensor] = None,
        high: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply FGSM to a single observation.

        Notes
        -----
        - Uses context.epsilon as the perturbation budget.
        - Returns a torch.Tensor (evaluation code can convert if needed).
        - 'low' and 'high' are optional bounds; pass them from the evaluator
          if you have access to env.observation_space bounds.
        """
        pol = self._check_policy(policy)

        eps = float(context.epsilon)
        if eps <= 0.0:
            # No attack requested; return observation as tensor for consistency.
            return torch.as_tensor(observation, dtype=torch.float32)

        # Convert to tensor and enable gradient tracking.
        x = torch.as_tensor(observation, dtype=torch.float32)
        x = x.detach().clone().requires_grad_(True)

        # Compute scalar objective.
        if self.objective_fn is not None:
            loss = self.objective_fn(x)
        else:
            loss = pol.loss(x)

        # Compute gradient w.r.t. observation.
        loss.backward()
        grad = x.grad
        if grad is None:
            raise RuntimeError("Gradient is None. Policy/loss might be non-differentiable.")

        # Take one FGSM step.
        if self.norm == "linf":
            x_adv = x + eps * grad.sign()
            x_adv = project_linf(x_adv, x.detach(), eps)
        else:
            # For L2 FGSM, normalize gradient direction.
            g = grad.view(-1)
            g_norm = torch.norm(g) + 1e-12
            direction = (grad / g_norm)
            x_adv = x + eps * direction
            x_adv = project_l2(x_adv, x.detach(), eps)

        # Optional clipping to valid observation bounds.
        if self.clip_bounds and (low is not None) and (high is not None):
            x_adv = clip_to_bounds(x_adv, low=low, high=high)

        return x_adv.detach()
