"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Base policy interfaces.

This module defines the minimal contracts that a policy must
satisfy to be used within RoAd-RL. The separation between
action-only and differentiable policies is intentional and
allows the library to support both black-box and gradient-based
attacks cleanly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Policy(ABC):
    """
    Minimal policy interface.

    Any policy used for evaluation must be able to take an
    observation and return an action. This interface is sufficient
    for clean evaluation and black-box attacks.
    """

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        Compute an action from an observation.

        This method must not modify internal policy state.
        """
        raise NotImplementedError


class DifferentiablePolicy(Policy):
    """
    Extension of the Policy interface for gradient-based attacks.

    Policies implementing this interface must expose a differentiable
    computation graph so that gradients with respect to the observation
    can be computed.
    """

    @abstractmethod
    def forward(self, observation: Any) -> Any:
        """
        Forward pass that returns raw policy outputs (e.g. logits
        or distribution parameters) while preserving gradients.
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, observation: Any) -> Any:
        """
        Scalar loss used by adversarial attacks.

        Typical choices include negative log-probability of the
        selected action or a value-function-based objective.
        """
        raise NotImplementedError
