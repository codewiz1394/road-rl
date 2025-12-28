"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Eval CLI subcommand.

This module wires the `road-rl eval` CLI command to the underlying
evaluation pipeline (env creation, sweep runner, logging, metrics,
and plotting). No evaluation logic lives here.
"""

from __future__ import annotations

import argparse
import json
import importlib
from typing import Any

from road_rl.envs.make_env import EnvSpec, make_env_factory
from road_rl.eval.sweep_runner import run_sweep
from road_rl.io.logger import save_experiment
from road_rl.io.plotting import (
    plot_return_vs_epsilon,
    plot_violin_returns,
    plot_raw_rewards,
)
from road_rl.io.results import make_result_paths
from road_rl.metrics.robustness import compute_metrics_from_csv
from road_rl.attacks.fgsm import FGSMAttack
from road_rl.attacks.pgd import PGDAttack
from road_rl.attacks.jsma import JSMAAttack
from road_rl.defenses.normalize_clip import NormalizeClipDefense
from road_rl.policies.base import Policy


# ---------------------------------------------------------------------
# Argument handling
# ---------------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Register arguments for `road-rl eval`.
    """
    # Environment
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--adapter", default="gym", choices=["gym", "highway", "atari"])
    parser.add_argument("--render-mode", default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)

    # Policy
    parser.add_argument(
        "--policy-loader",
        required=True,
        help="Callable in the form module.path:function returning a Policy",
    )
    parser.add_argument("--policy-path", default=None)
    parser.add_argument("--algorithm", required=True)

    # Sweep
    parser.add_argument("--eps", nargs="+", type=float, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--episodes-per-seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)

    # Attack / defense
    parser.add_argument("--attack", default="none",
                        choices=["none", "fgsm", "pgd", "jsma"])
    parser.add_argument("--defense", default="none",
                        choices=["none", "normalize_clip"])
    parser.add_argument("--norm", default="linf", choices=["linf", "l2"])
    parser.add_argument("--attack-steps", type=int, default=10)
    parser.add_argument("--attack-step-size", type=float, default=None)
    parser.add_argument("--jsma-top-k", type=int, default=1)

    # Results / output
    parser.add_argument("--out", required=True, help="Root results directory")
    parser.add_argument("--experiment", required=True,
                        help="User-defined experiment name (top-level folder)")

    # Plotting
    parser.add_argument("--sma-window", type=int, default=None)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_callable(path: str) -> Any:
    if ":" not in path:
        raise ValueError("Expected format module.path:function_name")

    module_path, fn_name = path.split(":", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Callable '{fn_name}' not found in '{module_path}'")
    return fn


def _build_policy(loader_path: str, policy_path: str | None) -> Policy:
    loader = _load_callable(loader_path)
    policy = loader(policy_path) if policy_path else loader()

    if not isinstance(policy, Policy):
        raise TypeError(
            "Policy loader did not return a road_rl.policies.base.Policy instance."
        )
    return policy


def _build_attack(args) -> Any:
    if args.attack == "none":
        return None
    if args.attack == "fgsm":
        return FGSMAttack(norm=args.norm)
    if args.attack == "pgd":
        return PGDAttack(
            steps=args.attack_steps,
            step_size=args.attack_step_size,
            norm=args.norm,
            random_start=True,
        )
    if args.attack == "jsma":
        return JSMAAttack(
            steps=args.attack_steps,
            step_size=args.attack_step_size or 0.01,
            top_k=args.jsma_top_k,
            norm=args.norm,
        )
    raise ValueError(f"Unknown attack {args.attack}")


def _build_defense(args) -> Any:
    if args.defense == "none":
        return None
    if args.defense == "normalize_clip":
        return NormalizeClipDefense()
    raise ValueError(f"Unknown defense {args.defense}")


# ---------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    """
    Execute the eval command.
    """

    # Environment factory
    spec = EnvSpec(
        env_id=args.env_id,
        adapter=args.adapter,
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
    )
    env_factory = make_env_factory(spec)

    # Policy
    policy = _build_policy(args.policy_loader, args.policy_path)

    # Attack / defense
    attack = _build_attack(args)
    defense = _build_defense(args)

    # Run sweep
    result = run_sweep(
        env_factory=env_factory,
        policy=policy,
        env_id=args.env_id,
        algorithm=args.algorithm,
        epsilons=args.eps,
        seeds=args.seeds,
        attack=attack,
        defense=defense,
        episodes_per_seed=args.episodes_per_seed,
        max_steps=args.max_steps,
        metadata={
            "adapter": args.adapter,
            "attack": args.attack,
            "defense": args.defense,
            "norm": args.norm,
        },
    )

    # Construct result paths (THIS IS THE FIX)
    paths = make_result_paths(
        root_dir=args.out,
        experiment_name=args.experiment,
        env_id=args.env_id,
        algorithm=args.algorithm,
        attack=result.attack_name,
        defense=result.defense_name,
    )

    # Save raw experiment (episodes.csv + meta.json)
    save_experiment(result, paths.run_dir)

    # Metrics
    metrics = compute_metrics_from_csv(str(paths.episodes_csv))
    paths.metrics_json.write_text(json.dumps(metrics, indent=2))

    # Plots
    plot_return_vs_epsilon(
        csv_files=[paths.episodes_csv],
        labels=[f"{args.attack}/{args.defense}"],
        output_path=paths.return_vs_eps_pdf,
        sma_window=args.sma_window,
    )

    plot_violin_returns(
        csv_files=[paths.episodes_csv],
        labels=[f"{args.attack}/{args.defense}"],
        output_path=paths.violin_pdf,
    )

    plot_raw_rewards(
        csv_file=paths.episodes_csv,
        output_path=paths.raw_rewards_pdf,
    )

    print(f"[RoAd-RL] Evaluation complete.")
    print(f"[RoAd-RL] Results written to: {paths.run_dir}")
