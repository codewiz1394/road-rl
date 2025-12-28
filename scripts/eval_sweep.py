#!/usr/bin/env python3
"""
RoAd-RL evaluation sweep runner (script).

This script runs a robustness sweep over epsilons and seeds, saves results,
computes metrics, and generates paper-ready plots.

Policies:
- This script is policy-framework agnostic. It expects a Policy adapter
  implementing road_rl.policies.base.Policy.
- You will plug your minimal RL policies via the --policy-module option.

Example:
  python scripts/eval_sweep.py \
    --env-id CartPole-v1 --adapter gym --algorithm minimal-ppo \
    --policy-module my_minrl.policies.cartpole:load_policy \
    --attack fgsm --eps 0.0 0.01 0.02 0.05 --seeds 0 1 2 \
    --out runs/cartpole_fgsm

"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from road_rl.envs.make_env import EnvSpec, make_env_factory
from road_rl.eval.sweep_runner import run_sweep
from road_rl.io.logger import save_experiment
from road_rl.io.plotting import (
    plot_return_vs_epsilon,
    plot_violin_returns,
    plot_raw_rewards,
)
from road_rl.metrics.robustness import compute_metrics_from_csv
from road_rl.policies.base import Policy

from road_rl.attacks.fgsm import FGSMAttack
from road_rl.attacks.pgd import PGDAttack
from road_rl.attacks.jsma import JSMAAttack
from road_rl.defenses.normalize_clip import NormalizeClipDefense


# ------------------------------------------------------------
# Policy loading utilities
# ------------------------------------------------------------

def load_callable(path: str) -> Callable[..., Any]:
    """
    Load a callable from 'module.submodule:function_name'.
    """
    if ":" not in path:
        raise ValueError("Expected --policy-module in the form 'module.path:function_name'.")

    mod_path, fn_name = path.split(":", 1)
    module = importlib.import_module(mod_path)
    fn = getattr(module, fn_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Could not load callable '{fn_name}' from module '{mod_path}'.")
    return fn


def build_policy(policy_loader: Callable[..., Any], policy_path: Optional[str]) -> Policy:
    """
    Build a Policy instance.

    The loader must return an object implementing road_rl.policies.base.Policy.
    """
    if policy_path is None:
        policy = policy_loader()
    else:
        policy = policy_loader(policy_path)

    if not isinstance(policy, Policy):
        raise TypeError(
            "Loaded policy does not implement road_rl.policies.base.Policy. "
            f"Got: {type(policy).__name__}"
        )
    return policy


# ------------------------------------------------------------
# Component builders
# ------------------------------------------------------------

def build_attack(name: str, *, norm: str, steps: int, step_size: Optional[float], top_k: int) -> Any:
    name = name.lower()

    if name == "none" or name == "clean":
        return None

    if name == "fgsm":
        return FGSMAttack(norm=norm)

    if name == "pgd":
        return PGDAttack(steps=steps, step_size=step_size, norm=norm, random_start=True)

    if name == "jsma":
        return JSMAAttack(steps=steps, step_size=step_size or 0.01, top_k=top_k, norm=norm)

    raise ValueError(f"Unknown attack '{name}'. Use one of: none, fgsm, pgd, jsma.")


def build_defense(name: str) -> Any:
    name = name.lower()

    if name == "none":
        return None

    if name == "normalize_clip":
        # For now this is a no-statistics baseline unless user passes mean/std via config later.
        return NormalizeClipDefense()

    raise ValueError(f"Unknown defense '{name}'. Use one of: none, normalize_clip.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # Environment
    p.add_argument("--env-id", required=True, type=str)
    p.add_argument("--adapter", default="gym", choices=["gym", "highway", "atari"])
    p.add_argument("--render-mode", default=None, type=str)
    p.add_argument("--max-episode-steps", default=None, type=int)

    # Policy
    p.add_argument("--algorithm", required=True, type=str)
    p.add_argument("--policy-module", required=True, type=str,
                   help="Callable loader: 'module.path:function_name' returning a Policy.")
    p.add_argument("--policy-path", default=None, type=str,
                   help="Optional path passed to the policy loader (checkpoint).")

    # Sweep
    p.add_argument("--eps", nargs="+", required=True, type=float, help="List of epsilons.")
    p.add_argument("--seeds", nargs="+", required=True, type=int, help="List of seeds.")
    p.add_argument("--episodes-per-seed", default=1, type=int)
    p.add_argument("--max-steps", default=None, type=int)

    # Attack + defense
    p.add_argument("--attack", default="none", choices=["none", "fgsm", "pgd", "jsma"])
    p.add_argument("--defense", default="none", choices=["none", "normalize_clip"])
    p.add_argument("--norm", default="linf", choices=["linf", "l2"])
    p.add_argument("--attack-steps", default=10, type=int, help="PGD/JSMA steps.")
    p.add_argument("--attack-step-size", default=None, type=float, help="PGD/JSMA step size.")
    p.add_argument("--jsma-top-k", default=1, type=int)

    # Output
    p.add_argument("--out", required=True, type=str, help="Output directory.")
    p.add_argument("--prefix", default=None, type=str, help="Optional filename prefix.")

    # Plotting
    p.add_argument("--sma-window", default=None, type=int,
                   help="SMA window for robustness curves. If omitted, adaptive default is used.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build env factory
    spec = EnvSpec(
        env_id=args.env_id,
        render_mode=args.render_mode,
        max_episode_steps=args.max_episode_steps,
        adapter=args.adapter,
    )
    env_factory = make_env_factory(spec)

    # Load policy
    policy_loader = load_callable(args.policy_module)
    policy = build_policy(policy_loader, args.policy_path)

    # Build attack/defense
    attack = build_attack(
        args.attack,
        norm=args.norm,
        steps=args.attack_steps,
        step_size=args.attack_step_size,
        top_k=args.jsma_top_k,
    )
    defense = build_defense(args.defense)

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
            "attack_steps": args.attack_steps,
            "attack_step_size": args.attack_step_size,
            "jsma_top_k": args.jsma_top_k,
        },
    )

    # Save results
    save_experiment(result, out_dir, prefix=args.prefix)

    # Locate written episode CSV (same naming scheme as logger.py)
    base_parts = [args.env_id, args.algorithm, result.attack_name or "clean", result.defense_name or "none"]
    if args.prefix:
        base_parts.insert(0, args.prefix)
    base_name = "_".join(base_parts)
    csv_path = out_dir / f"{base_name}_episodes.csv"

    # Compute and store metrics
    metrics = compute_metrics_from_csv(str(csv_path), cvar_alpha=0.1)
    metrics_path = out_dir / f"{base_name}_metrics.json"
    metrics_path.write_text(__import__("json").dumps(metrics, indent=2))

    # Generate plots
    plot_return_vs_epsilon(
        csv_files=[csv_path],
        labels=[f"{args.attack}/{args.defense}"],
        output_path=out_dir / f"{base_name}_return_vs_eps.pdf",
        title=f"{args.env_id} ({args.algorithm})",
        sma_window=args.sma_window,
    )

    plot_violin_returns(
        csv_files=[csv_path],
        labels=[f"{args.attack}/{args.defense}"],
        output_path=out_dir / f"{base_name}_violin_returns.pdf",
        title=f"{args.env_id} returns",
    )

    plot_raw_rewards(
        csv_file=csv_path,
        output_path=out_dir / f"{base_name}_raw_rewards.pdf",
        title=f"{args.env_id} episodic returns",
    )

    print(f"[RoAd-RL] Done. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
