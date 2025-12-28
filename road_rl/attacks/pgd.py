"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

PGD (Projected Gradient Descent) attack implementation.

PGD is an iterative gradient-based attack:
    x_{k+1} = Project_{B(x0, eps)}( x_k + alpha * sign(grad_x loss(x_k)) )

This module implements observation-space PGD with optional
random start. Action-space variants can be added later using
a similar interface.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Literal

import torch

from road_rl.attacks.base import GradientAttack
from road_rl.attacks.constraints import project_linf, project_l2, clip_to_bounds
from road_rl.core.types import StepContext
from road_rl.policies.base import Policy


NormType = Literal["linf", "l2"]


class PGDAttack(GradientAttack):
    """
    PGD observation attack.

    Parameters
    ----------
    steps:
        Number of PGD iterations.

    step_size:
        Per-iteration step size (alpha). If None, a common default
        is used: alpha = epsilon / steps.

    norm:
        Norm constraint used for the perturbation ("linf" or "l2").

    random_start:
        If True, start from a random point within the epsilon ball.

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
        step_size: Optional[float] = None,
        norm: NormType = "linf",
        random_start: bool = True,
        clip_bounds: bool = True,
        objective_fn: Optional[Callable[[Any], torch.Tensor]] = None,
    ):
        if steps <= 0:
            raise ValueError("steps must be a positive integer.")
        if norm not in ("linf", "l2"):
            raise ValueError(f"Unsupported norm '{norm}'. Use 'linf' or 'l2'.")

        self.steps = int(steps)
        self.step_size = step_size
        self.norm = norm
        self.random_start = bool(random_start)
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
        Apply PGD to a single observation.

        Notes
        -----
        - Uses context.epsilon as the perturbation budget.
        - Returns a torch.Tensor (detached).
        """
        pol = self._check_policy(policy)

        eps = float(context.epsilon)
        if eps <= 0.0:
            return torch.as_tensor(observation, dtype=torch.float32)

        alpha = self.step_size
        if alpha is None:
            alpha = eps / float(self.steps)

        x0 = torch.as_tensor(observation, dtype=torch.float32).detach()

        # Initialize within the epsilon ball if random_start is enabled.
        if self.random_start:
            if self.norm == "linf":
                delta = torch.empty_like(x0).uniform_(-eps, eps)
                x = x0 + delta
                x = project_linf(x, x0, eps)
            else:
                # Sample random direction and scale to within L2 ball.
                delta = torch.randn_like(x0)
                delta_flat = delta.view(-1)
                delta_norm = torch.norm(delta_flat) + 1e-12
                # Uniform radius in [0, eps]
                radius = torch.rand(1, device=x0.device).item() * eps
                delta = delta * (radius / delta_norm)
                x = x0 + delta
                x = project_l2(x, x0, eps)
        else:
            x = x0.clone()

        # Optionally clip initial point to bounds.
        if self.clip_bounds and (low is not None) and (high is not None):
            x = clip_to_bounds(x, low=low, high=high)

        # PGD loop
        for _ in range(self.steps):
            x = x.detach().clone().requires_grad_(True)

            if self.objective_fn is not None:
                loss = self.objective_fn(x)
            else:
                loss = pol.loss(x)

            loss.backward()
            grad = x.grad
            if grad is None:
                raise RuntimeError("Gradient is None. Policy/loss might be non-differentiable.")

            if self.norm == "linf":
                x = x + float(alpha) * grad.sign()
                x = project_linf(x, x0, eps)
            else:
                g = grad.view(-1)
                g_norm = torch.norm(g) + 1e-12
                direction = grad / g_norm
                x = x + float(alpha) * direction
                x = project_l2(x, x0, eps)

            if self.clip_bounds and (low is not None) and (high is not None):
                x = clip_to_bounds(x, low=low, high=high)

        return x.detach()
