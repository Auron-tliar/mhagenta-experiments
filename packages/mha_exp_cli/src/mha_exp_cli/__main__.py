import argparse
from pathlib import Path
from collections.abc import Sequence

from .cli import available_experiments, parse_run_range, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run installed MHAgentA experiments."
    )
    parser.add_argument(
        "experiment",
        type=str,
        nargs="?",
        help="Experiment name, e.g. 1.1, 2.bw.1, or 2.cr.1.",
    )
    runs_group = parser.add_mutually_exclusive_group()
    runs_group.add_argument(
        "-n",
        "--num-runs",
        type=non_negative_int,
        default=None,
        help="Number of repeated runs to execute; defaults to the experiment's setting.",
    )
    runs_group.add_argument(
        "-r",
        "--run-range",
        type=run_range,
        default=None,
        help="Run numbers to execute, e.g. 1,4,7-10.",
    )
    parser.add_argument(
        "-w",
        "--work-dir",
        type=Path,
        default=None,
        help="Directory for agent runtime files and saved states.",
    )
    parser.add_argument(
        "--mha-version",
        type=non_empty_string,
        default=None,
        help="MHAgentA version to use; defaults to the selected experiment's version.",
    )
    parser.add_argument(
        "--process-only",
        action="store_true",
        help="Skip execution and process existing results in the work directory.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List installed experiments and exit.",
    )
    parser.add_argument(
        "--config-file", type=Path,
        help="JSON object of additional options accepted by the experiment's batch runner.",
    )
    return parser


def non_negative_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer: {value!r}") from exc

    if result < 0:
        raise argparse.ArgumentTypeError("value cannot be negative")
    return result


def run_range(value: str) -> list[int]:
    try:
        return parse_run_range(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def non_empty_string(value: str) -> str:
    if not value:
        raise argparse.ArgumentTypeError("value cannot be empty")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    experiments = available_experiments()
    if args.list:
        for name in sorted(experiments):
            print(name)
        return 0

    if args.experiment is None:
        parser.error("the following arguments are required: experiment")

    if args.experiment not in experiments:
        available = ", ".join(sorted(experiments)) or "none"
        parser.error(f"unknown experiment {args.experiment!r}. Available: {available}")

    runs = args.run_range if args.run_range is not None else args.num_runs
    run_experiment(
        name=args.experiment,
        runs=runs,
        work_dir=args.work_dir,
        mha_version=args.mha_version,
        process_only=args.process_only,
        config_file=args.config_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
