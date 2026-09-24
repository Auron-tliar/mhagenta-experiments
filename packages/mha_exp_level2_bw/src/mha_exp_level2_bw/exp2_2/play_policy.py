from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import numpy as np

from .policy import (
    NUM_BLOCKS,
    POLICY_FILENAME,
    TABLE_LEN,
    as_numeric_observation,
    goal_achieved,
    goal_conditioned_observation,
    load_policy_checkpoint,
    sample_goal,
    legal_action_mask,
)


DEFAULT_FPS = 2.0
DEFAULT_MAX_STEPS = 200
PlaybackMode = Literal["gif", "human"]


@dataclass(frozen=True)
class EpisodeResult:
    success: bool
    steps: int
    window_closed: bool = False


def require_torch() -> Any:
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Torch is required for policy playback. Install the optional extra with "
            "`uv sync --extra playback` or install this package as "
            "`mha-exp-level2-bw[playback]`."
        ) from exc


def greedy_action(
    torch_module: Any,
    model: Any,
    observation: np.ndarray,
    goal: tuple[int, int],
    *, action_masking: bool = False,
) -> int:
    model_input = goal_conditioned_observation(observation, goal)
    tensor = torch_module.as_tensor(
        model_input,
        dtype=torch_module.float32,
    ).unsqueeze(0)
    with torch_module.no_grad():
        values = model(tensor)
        if action_masking:
            valid = torch_module.as_tensor(legal_action_mask(observation), device=values.device)
            values = values.masked_fill(~valid, -torch_module.inf)
        return int(values.argmax(dim=1).item())


def evaluate_episode(
    torch_module: Any,
    model: Any,
    environment: Any,
    observation: np.ndarray,
    goal: tuple[int, int],
    *,
    max_steps: int,
    mode: PlaybackMode,
    fps: float,
    frame_writer: Any | None = None,
    action_masking: bool = False,
) -> EpisodeResult:
    if frame_writer is not None:
        frame_writer.append_data(environment.render())

    for step in range(1, max_steps + 1):
        started = time.monotonic()
        action = (greedy_action(torch_module, model, observation, goal, action_masking=True)
                  if action_masking else greedy_action(torch_module, model, observation, goal))
        observation, _, _, _, _ = environment.step(action)
        observation = as_numeric_observation(observation)

        if frame_writer is not None:
            frame_writer.append_data(environment.render())

        if mode == "human":
            elapsed = time.monotonic() - started
            time.sleep(max(0.0, 1.0 / fps - elapsed))
            if environment.window is None:
                return EpisodeResult(False, step, window_closed=True)

        if goal_achieved(observation, goal):
            return EpisodeResult(True, step)

    return EpisodeResult(False, max_steps)


def _episode_filename(episode: int, goal: tuple[int, int]) -> str:
    digits = len(str(NUM_BLOCKS - 1))
    return (
        f"episode_{episode:03d}_"
        f"B{goal[0]:0{digits}d}_on_B{goal[1]:0{digits}d}.gif"
    )


def run_playback(
    checkpoint_path: str | Path,
    *,
    episodes: int,
    mode: PlaybackMode = "gif",
    fps: float = DEFAULT_FPS,
    max_steps: int = DEFAULT_MAX_STEPS,
    seed: int = 0,
    output_dir: str | Path | None = None,
) -> list[Path]:
    if episodes <= 0:
        raise ValueError("episodes must be positive.")
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    resolved_checkpoint = Path(checkpoint_path).resolve()
    torch_module = require_torch()
    environment_module = import_module("mha_env_blocksworld")
    imageio = import_module("imageio.v2") if mode == "gif" else None
    model, checkpoint = load_policy_checkpoint(torch_module, resolved_checkpoint)
    print(
        f"Loaded {resolved_checkpoint} "
        f"({checkpoint['training_steps']} training steps)."
    )

    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else resolved_checkpoint.parent / "policy_evaluation"
    )
    if mode == "gif":
        destination.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    environment = environment_module.BlocksWorldEnv(
        table_len=TABLE_LEN,
        num_blocks=NUM_BLOCKS,
        render_mode="rgb_array" if mode == "gif" else "human",
        symbolic=False,
    )
    generated: list[Path] = []
    try:
        for episode in range(1, episodes + 1):
            observation, _ = environment.reset(seed=seed + episode - 1)
            observation = as_numeric_observation(observation)
            goal = sample_goal(rng, observation)
            print(
                f"Episode {episode}/{episodes}: "
                f"place block B{goal[0]:02d} on block B{goal[1]:02d}."
            )

            writer = None
            gif_path: Path | None = None
            try:
                if mode == "gif":
                    assert imageio is not None
                    gif_path = destination / _episode_filename(episode, goal)
                    writer = imageio.get_writer(
                        gif_path,
                        mode="I",
                        duration=1000.0 / fps,
                        loop=0,
                    )
                result = evaluate_episode(
                    torch_module,
                    model,
                    environment,
                    observation,
                    goal,
                    max_steps=max_steps,
                    mode=mode,
                    fps=fps,
                    frame_writer=writer,
                    action_masking=checkpoint.get("training_protocol", {}).get("config", {}).get("action_masking", False),
                )
            finally:
                if writer is not None:
                    writer.close()

            outcome = "success" if result.success else "step limit reached"
            print(f"Episode {episode}: {outcome} after {result.steps} steps.")
            if gif_path is not None:
                gif_path = gif_path.resolve()
                generated.append(gif_path)
                print(f"GIF: {gif_path}")
            if result.window_closed:
                print("Playback window closed; stopping evaluation.")
                break
    finally:
        environment.close()

    return generated


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved 2-2-BW DQN policy without exploration."
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help=f"Path to a saved {POLICY_FILENAME} checkpoint.",
    )
    parser.add_argument("--episodes", type=_positive_int, required=True)
    parser.add_argument(
        "--mode",
        choices=("gif", "human"),
        default="gif",
    )
    parser.add_argument("--fps", type=_positive_float, default=DEFAULT_FPS)
    parser.add_argument(
        "--max-steps",
        type=_positive_int,
        default=DEFAULT_MAX_STEPS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_playback(
            args.checkpoint,
            episodes=args.episodes,
            mode=args.mode,
            fps=args.fps,
            max_steps=args.max_steps,
            seed=args.seed,
            output_dir=args.output_dir,
        )
    except KeyboardInterrupt:
        print("Playback interrupted.")
        return 130
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Playback failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
