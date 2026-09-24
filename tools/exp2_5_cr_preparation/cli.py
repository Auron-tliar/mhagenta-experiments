"""Command-line interface for independent Crafter policy preparation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

POLICIES = ("explore", "navigate_to", "get_resource", "eat_target", "eat_cow")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse training, diagnostic, evaluation, or five-model export options."""

    parser = argparse.ArgumentParser(description="Train and use the current five Crafter policies.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "train"):
        command = commands.add_parser(name)
        command.add_argument("directory", type=Path)
        command.add_argument("--device", choices=("cpu", "cuda"), required=True)
        command.add_argument("--policy", choices=POLICIES)
        command.add_argument("--init-checkpoint", type=Path)
        if name == "train":
            command.add_argument("--resume", action="store_true")
    command = commands.add_parser("evaluate")
    command.add_argument("checkpoint", type=Path)
    command.add_argument("--policy", choices=POLICIES, required=True)
    command.add_argument("--device", choices=("cpu", "cuda"), required=True)
    command.add_argument("--seed", type=int, default=5_240_000)
    command.add_argument("--report", type=Path, required=True)
    command = commands.add_parser("export")
    command.add_argument("destination", type=Path)
    command.add_argument("--checkpoint", action="append", required=True, metavar="POLICY=PATH")
    return parser.parse_args(argv)
