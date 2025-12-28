"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Evaluation orchestrator.

This module provides a higher-level evaluation interface on top of the
sweep runner. It handles seeding, reproducibility, and coordination
of multiple sweeps without duplicating execution logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from road_rl.core.seeding import seed_everything
from road_rl.eval.sweep_runner import run_sweep
from road_rl.core.types import ExperimentResult
from road_rl.attacks.base import Attack
from road_rl.defenses.base import Defense
from road_rl.policies.base import Policy


# ---------------------------------------------------------------------
# Evaluation specification
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class EvalSpec:
    """
    Specification for a single evaluation sweep.
    """
    env_factory: Any
    env_id: str
    algorithm: str
    epsilons: Sequence[float]
    seeds: Sequence[int]
    attack: Optional[Attack] = None
    defense: Optional[Defense] = None
    episodes_per_seed: int = 1
    max_steps: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------

class Evaluator:
    """
    Evaluation orchestrator.

    The Evaluator coordinates one or more evaluation sweeps under
    controlled seeding conditions. It is intentionally stateless
    across calls to `run` to avoid hidden interactions.
    """

    def __init__(
        self,
        *,
        base_seed: Optional[int] = None,
        deterministic_torch: bool = True,
    ):
        """
        Parameters
        ----------
        base_seed:
            Optional base seed applied before running evaluations.

        deterministic_torch:
            If True, configure PyTorch for deterministic behavior.
        """
        self._base_seed = base_seed
        self._deterministic_torch = deterministic_torch

    def run(
        self,
        policy: Policy,
        specs: Iterable[EvalSpec],
    ) -> List[ExperimentResult]:
        """
        Run one or more evaluation sweeps.

        Parameters
        ----------
        policy:
            Frozen policy to evaluate.

        specs:
            Iterable of EvalSpec objects defining each sweep.

        Returns
        -------
        List[ExperimentResult]
            Results for each evaluation sweep.
        """
        if self._base_seed is not None:
            seed_everything(
                self._base_seed,
                deterministic_torch=self._deterministic_torch,
            )

        results: List[ExperimentResult] = []

        for spec in specs:
            result = run_sweep(
                env_factory=spec.env_factory,
                policy=policy,
                env_id=spec.env_id,
                algorithm=spec.algorithm,
                epsilons=spec.epsilons,
                seeds=spec.seeds,
                attack=spec.attack,
                defense=spec.defense,
                episodes_per_seed=spec.episodes_per_seed,
                max_steps=spec.max_steps,
                metadata=spec.metadata,
            )
            results.append(result)

        return results
