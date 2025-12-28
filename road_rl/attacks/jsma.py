"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

JSMA (Jacobian-based Saliency Map Attack) implementation.

This is a feature-selective, gradient-based attack. Unlike FGSM/PGD,
which perturb all dimensions, JSMA perturbs only the most influential
features according to a saliency score derived from gradients.

This v1 implementation focuses on observation-space, untargeted JSMA.
It is most effective and interpretable for low-dimensional state spaces.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Literal

import torch

from road_rl.attacks.base import GradientAttack
from road_rl.attacks.constraints import project_linf, project_l2, clip_to_bounds
from road_rl.core.types import StepContext
from road_rl.policies.base import Policy


NormType = Literal["linf", "l2"]


class JSMAAttack(GradientAttack):
    """
    JSMA observation attack (untargeted, feature-selective).

    Parameters
    ----------
    steps:
        Number of JSMA iterations.

    step_size:
        Magnitude applied to selected features each step.

    top_k:
        Number of features to modify per step. For small state vectors,
        values like 1 or 2 are typical.

    norm:
        Norm constraint used for the perturbation ("linf" or "l2").

    clip_bounds:
        If True, clip attacked observations to [low, high] bounds
        when bounds are provided.

    objective_fn:
        Optional custom objective. If not provided, policy.loss(obs)
        is used.
    """

    def __init__(
        self,
        steps: int = 10,
        step_size: float = 0.01,
        top_k: int = 1,
        norm: NormType = "linf",
        clip_bounds: bool = True,
        objective_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        if step_size <= 0:
            raise ValueError("step_size must be > 0.")
        if norm not in ("linf", "l2"):
            raise ValueError(f"Unsupported norm '{norm}'. Use 'linf' or 'l2'.")

        self.steps = int(steps)
        self.step_size = float(step_size)
        self.top_k = int(top_k)
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
        Apply JSMA to a single observation.

        Notes
        -----
        - Uses context.epsilon as the perturbation budget.
        - Returns a torch.Tensor (detached).
        """
        pol = self._check_policy(policy)

        eps = float(context.epsilon)
        if eps <= 0.0:
            return torch.as_tensor(observation, dtype=torch.float32)

        x0 = torch.as_tensor(observation, dtype=torch.float32).detach()
        x = x0.clone()

        # Optionally clip initial point to bounds.
        if self.clip_bounds and (low is not None) and (high is not None):
            x = clip_to_bounds(x, low=low, high=high)

        # JSMA loop
        for _ in range(self.steps):
            x_var = x.detach().clone().requires_grad_(True)

            if self.objective_fn is not None:
                loss = self.objective_fn(x_var)
            else:
                loss = pol.loss(x_var)

            loss.backward()
            grad = x_var.grad
            if grad is None:
                raise RuntimeError("Gradient is None. Policy/loss might be non-differentiable.")

            # Flatten gradients to rank features.
            grad_flat = grad.view(-1)
            # Saliency = absolute gradient magnitude (simple, robust baseline)
            saliency = torch.abs(grad_flat)

            k = min(self.top_k, saliency.numel())
            idx = torch.topk(saliency, k=k, largest=True).indices

            # Apply sparse perturbation in the direction of the gradient.
            delta = torch.zeros_like(grad_flat)
            delta[idx] = torch.sign(grad_flat[idx]) * self.step_size

            x = x + delta.view_as(x)

            # Project back to epsilon ball around x0.
            if self.norm == "linf":
                x = project_linf(x, x0, eps)
            else:
                x = project_l2(x, x0, eps)

            # Optional clipping to valid observation bounds.
            if self.clip_bounds and (low is not None) and (high is not None):
                x = clip_to_bounds(x, low=low, high=high)

        return x.detach()
