"""Execute a fresh bounded eight-module CUDA preflight before the one-hour main CLI."""

import argparse
import json
from pathlib import Path

from mha_exp_level2_cr.exp2_6.online_runner import run_experiment


def main() -> None:
    """Use main learning settings and environment limits in a separately labelled attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if not 120 <= args.seconds <= 900 or args.gpu_id < 0:
        parser.error("Use a 120–900-second preflight and a nonnegative GPU ID")
    result = run_experiment(0, args.output.resolve(), duration_seconds=args.seconds,
                            device="cuda", gpu_id=args.gpu_id)
    print(json.dumps(result, allow_nan=False))
    if not result["execution_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
