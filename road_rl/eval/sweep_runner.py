"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Sweep execution logic.

This module runs repeated evaluation episodes across a list of
epsilon values and seeds, collecting results into a structured
ExperimentResult. This is the canonical entry point for robustness
benchmarking experiments.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, List, Callable

from road_rl.core.types import EpisodeResult, ExperimentResult
from road_rl.eval.episode_runner import run_episode
from road_rl.policies.base import Policy
from road_rl.attacks.base import Attack
from road_rl.defenses.base import Defense


def run_sweep(
    env_factory: Callable[[], Any],
    policy: Policy,
    *,
    env_id: str,
    algorithm: str,
    epsilons: Sequence[float],
    seeds: Sequence[int],
    attack: Optional[Attack] = None,
    defense: Optional[Defense] = None,
    episodes_per_seed: int = 1,
    max_steps: Optional[int] = None,
    metadata: Optional[dict] = None,
) -> ExperimentResult:
    """
    Run a full evaluation sweep across epsilons and seeds.

    Parameters
    ----------
    env_factory:
        Callable that returns a fresh environment instance.

    policy:
        Frozen policy to evaluate.

    env_id:
        Environment identifier (e.g. "CartPole-v1").

    algorithm:
        Algorithm name used to train the policy (e.g. "ppo", "dqn", "sac").

    epsilons:
        Sequence of attack budgets to evaluate.

    seeds:
        Random seeds for reproducibility.

    attack:
        Optional adversarial attack.

    defense:
        Optional defense.

    episodes_per_seed:
        Number of episodes to run per (epsilon, seed) pair.

    max_steps:
        Optional maximum number of steps per episode.

    metadata:
        Optional dictionary stored with the experiment results.

    Returns
    -------
    ExperimentResult
        Aggregated experiment results.
    """
    if episodes_per_seed <= 0:
        raise ValueError("episodes_per_seed must be a positive integer.")

    episode_results: List[EpisodeResult] = []
    episode_idx = 0

    for eps in epsilons:
        eps = float(eps)

        for seed in seeds:
            seed = int(seed)

            for _ in range(episodes_per_seed):
                env = env_factory()

                result = run_episode(
                    env=env,
                    policy=policy,
                    episode=episode_idx,
                    attack=attack,
                    defense=defense,
                    epsilon=eps,
                    seed=seed,
                    max_steps=max_steps,
                )

                episode_results.append(result)
                episode_idx += 1

                try:
                    env.close()
                except Exception:
                    pass

    return ExperimentResult(
        env_id=env_id,
        algorithm=algorithm,
        attack_name=attack.__class__.__name__ if attack is not None else None,
        defense_name=defense.__class__.__name__ if defense is not None else None,
        epsilons=list(float(e) for e in epsilons),
        episode_results=episode_results,
        metadata=metadata or {},
    )
