"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Base attack interfaces.

This module defines what it means to be an adversarial attack
within RoAd-RL. Concrete attacks such as FGSM, PGD, or JSMA
must inherit from the Attack base class and implement the
required interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from road_rl.core.types import StepContext
from road_rl.policies.base import Policy, DifferentiablePolicy


class Attack(ABC):
    """
    Abstract base class for all adversarial attacks.

    An attack takes an observation and produces a perturbed
    observation. Attacks must be stateless and deterministic
    given the same inputs and random seed.
    """

    #: Whether this attack requires a differentiable policy
    requires_gradients: bool = False

    @abstractmethod
    def apply(
        self,
        observation: Any,
        policy: Policy,
        context: StepContext,
    ) -> Any:
        """
        Apply the attack to a single observation.

        Parameters
        ----------
        observation:
            The original environment observation.

        policy:
            The policy being evaluated. Some attacks may require
            this policy to be differentiable.

        context:
            Step-level metadata such as timestep index, epsilon,
            and environment info.

        Returns
        -------
        Any
            The perturbed observation to be passed to the policy.
        """
        raise NotImplementedError


class GradientAttack(Attack):
    """
    Base class for gradient-based attacks.

    This class enforces that the policy passed to the attack
    implements the DifferentiablePolicy interface.
    """

    requires_gradients: bool = True

    def _check_policy(self, policy: Policy) -> DifferentiablePolicy:
        """
        Ensure that the policy supports gradient-based attacks.
        """
        if not isinstance(policy, DifferentiablePolicy):
            raise TypeError(
                "This attack requires a DifferentiablePolicy, "
                f"but received {type(policy).__name__}."
            )
        return policy
