"""
RoAd-RL: Robust Adversarial Reinforcement Learning Library

Command-line interface (CLI).

This module provides the official `road-rl` command entry point.
It is intentionally thin: all heavy logic is delegated to
library modules or scripts to avoid duplication.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# We import the sweep script's main function logic explicitly.
# This avoids re-implementing evaluation logic in multiple places.
from road_rl.cli.subcommands.eval_cmd import run_eval_command


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level CLI argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="road-rl",
        description="RoAd-RL: Robust Adversarial Reinforcement Learning",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ---------------------------------------------------------
    # Eval subcommand
    # ---------------------------------------------------------
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run robustness evaluation sweeps",
    )

    run_eval_command.add_arguments(eval_parser)

    return parser


def main(argv: list[str] | None = None) -> None:
    """
    CLI entry point.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "eval":
        run_eval_command.run(args)
    else:
        # This should never happen due to argparse enforcement
        parser.error(f"Unknown command '{args.command}'")


if __name__ == "__main__":
    main(sys.argv[1:])
