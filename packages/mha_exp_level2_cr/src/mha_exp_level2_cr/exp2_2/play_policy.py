"""Optional greedy playback for a saved four-frame 2-2-CR policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import import_module
import json
from pathlib import Path
import time
from typing import Any, Literal, Sequence

import numpy as np

from .policy import (
    POLICY_FILENAME, TARGET_ACHIEVEMENT, as_rgb_frame, goal_achieved,
    initial_frame_stack, load_policy_checkpoint, shift_frame_stack, stack_batch,
)


DEFAULT_FPS = 2.0
DEFAULT_MAX_STEPS = 1_000
RENDER_SIZE = (512, 512)
PlaybackMode = Literal["gif", "human"]


@dataclass(frozen=True)
class EpisodeResult:
    """One playback episode outcome."""
    success: bool
    died: bool
    steps: int
    window_closed: bool = False


def require_torch() -> Any:
    """Import Torch with a playback-specific installation hint."""
    try:
        return import_module("torch")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Torch is required; install mha-exp-level2-cr[playback].") from exc


def greedy_action(torch_module: Any, model: Any, frame_stack: np.ndarray, *, mask: Any = None) -> int:
    """Return the greedy action for one canonical four-frame stack."""
    images = torch_module.as_tensor(stack_batch([frame_stack]), dtype=torch_module.float32) / 255.0
    with torch_module.no_grad():
        values = model(images)
        if mask is not None:
            from .masking import validate_mask
            valid = torch_module.as_tensor(validate_mask(mask), device=values.device)
            values = values.masked_fill(~valid, -torch_module.inf)
        return int(values.argmax(dim=1).item())


def _show_frame(pygame: Any, window: Any, frame: np.ndarray) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False
    window.blit(pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1)), (0, 0))
    pygame.display.flip()
    return True


def evaluate_episode(
    torch_module: Any, model: Any, environment: Any, observation: np.ndarray,
    *,
    max_steps: int,
    mode: PlaybackMode,
    fps: float,
    frame_writer: Any | None = None,
    pygame: Any | None = None,
    window: Any | None = None,
    action_masking: bool = False,
) -> EpisodeResult:
    """Run one fixed-target greedy episode with the training stack treatment."""
    frame_stack = initial_frame_stack(observation)
    frame = environment.render(RENDER_SIZE)
    if frame_writer is not None:
        frame_writer.append_data(frame)
    if pygame is not None and not _show_frame(pygame, window, frame):
        return EpisodeResult(False, False, 0, window_closed=True)

    for step in range(1, max_steps + 1):
        started = time.monotonic()
        from .masking import action_mask
        mask = action_mask(environment) if action_masking else None
        action = greedy_action(torch_module, model, frame_stack, mask=mask)
        observation, _, done, info = environment.step(action)
        dead = int(info["inventory"]["health"]) <= 0
        frame_stack = shift_frame_stack(frame_stack, as_rgb_frame(observation))
        frame = environment.render(RENDER_SIZE)
        if frame_writer is not None:
            frame_writer.append_data(frame)
        if pygame is not None and not _show_frame(pygame, window, frame):
            return EpisodeResult(False, dead, step, window_closed=True)
        if dead:
            return EpisodeResult(False, True, step)
        if goal_achieved(info.get("achievements")):
            return EpisodeResult(True, False, step)
        if done:
            return EpisodeResult(False, False, step)
        if mode == "human":
            time.sleep(max(0.0, 1.0 / fps - (time.monotonic() - started)))
    return EpisodeResult(False, False, max_steps)


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
    """Evaluate a saved policy as GIFs or in a human display window."""
    if episodes <= 0 or fps <= 0 or max_steps <= 0:
        raise ValueError("episodes, fps, and max_steps must be positive.")
    checkpoint = Path(checkpoint_path).resolve()
    torch_module = require_torch()
    environment_module = import_module("mha_env_crafter")
    imageio = import_module("imageio.v2") if mode == "gif" else None
    pygame = import_module("pygame") if mode == "human" else None
    model, metadata = load_policy_checkpoint(torch_module, checkpoint)
    print(f"Loaded {checkpoint} ({metadata['training_steps']} training steps).")
    print(f"Goal: {TARGET_ACHIEVEMENT.value.replace('_', ' ')}.")

    destination = Path(output_dir).resolve() if output_dir else checkpoint.parent / "policy_evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    environment = environment_module.CrafterEnv(seed=seed, length=10_000, symbolic=False, no_mobs=True)
    window = None
    if pygame is not None:
        pygame.init()
        window = pygame.display.set_mode(RENDER_SIZE)
        pygame.display.set_caption(f"Crafter policy: {TARGET_ACHIEVEMENT.value}")

    generated: list[Path] = []
    episode_rows: list[dict[str, Any]] = []
    try:
        for episode in range(1, episodes + 1):
            observation = as_rgb_frame(environment.reset())
            writer = None
            gif_path: Path | None = None
            try:
                if mode == "gif":
                    assert imageio is not None
                    gif_path = destination / f"episode_{episode:03d}_{TARGET_ACHIEVEMENT.value}.gif"
                    writer = imageio.get_writer(gif_path, mode="I", duration=1000.0 / fps, loop=0)
                result = evaluate_episode(
                    torch_module, model, environment, observation, max_steps=max_steps,
                    mode=mode, fps=fps, frame_writer=writer, pygame=pygame, window=window,
                    action_masking=metadata['workload']['action_masking'],
                )
            finally:
                if writer is not None:
                    writer.close()
            outcome = "success" if result.success else "death" if result.died else "step limit reached"
            print(f"Episode {episode}: {outcome} after {result.steps} steps.")
            if gif_path is not None:
                generated.append(gif_path.resolve())
                print(f"GIF: {gif_path.resolve()}")
            episode_rows.append({
                "episode": episode,
                "success": result.success,
                "death": result.died,
                "steps": result.steps,
                "window_closed": result.window_closed,
                "gif": None if gif_path is None else gif_path.name,
            })
            if result.window_closed:
                break
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()
        if pygame is not None:
            pygame.quit()
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    playback = {
        "schema_version": "2-2-cr-playback-v1",
        "protocol": {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_metadata": {
                key: value for key, value in metadata.items() if key != "model_state_dict"
            },
            "target_achievement": TARGET_ACHIEVEMENT.value,
            "seed": seed,
            "requested_episodes": episodes,
            "max_steps": max_steps,
            "mode": mode,
            "fps": fps,
            "crafter_length": 10_000,
            "symbolic": False,
        },
        "episodes": episode_rows,
    }
    (destination / "playback-results.json").write_text(
        json.dumps(playback, indent=2, sort_keys=True), encoding="utf-8"
    )
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
    """Build the playback command-line parser."""
    parser = argparse.ArgumentParser(description="Evaluate a saved four-frame 2-2-CR DQN policy.")
    parser.add_argument("checkpoint", type=Path, help=f"Path to {POLICY_FILENAME}.")
    parser.add_argument("--episodes", type=_positive_int, required=True)
    parser.add_argument("--mode", choices=("gif", "human"), default="gif")
    parser.add_argument("--fps", type=_positive_float, default=DEFAULT_FPS)
    parser.add_argument("--max-steps", type=_positive_int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run policy playback from the command line."""
    args = build_parser().parse_args(argv)
    try:
        run_playback(args.checkpoint, episodes=args.episodes, mode=args.mode,
                     fps=args.fps, max_steps=args.max_steps, seed=args.seed,
                     output_dir=args.output_dir)
    except KeyboardInterrupt:
        print("Playback interrupted.")
        return 130
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as exc:
        print(f"Playback failed: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
