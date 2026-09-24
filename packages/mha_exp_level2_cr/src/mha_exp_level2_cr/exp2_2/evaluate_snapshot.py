"""Evaluate an immutable periodic checkpoint in an independent CPU process."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def evaluate(checkpoint: Path, output: Path, *, timeout: float = 300.0) -> dict:
    """Evaluate fixed held-out seeds without any connection to training replay."""
    import torch
    from mha_env_crafter import CrafterEnv
    from .policy import load_policy_checkpoint, initial_frame_stack, shift_frame_stack, goal_achieved
    from .play_policy import greedy_action
    from .masking import action_mask

    torch.set_num_threads(1)
    started = time.monotonic()
    model, metadata = load_policy_checkpoint(torch, checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    rows = []
    deadline = started + timeout
    for seed in metadata['workload']['evaluation_seeds']:
        env = CrafterEnv(seed=seed, length=10000, symbolic=False, no_mobs=True)
        stack = initial_frame_stack(env.reset())
        rewards = 0.0
        illegal = 0
        histogram = [0] * 17
        success = dead = False
        for step in range(1, metadata['workload']['episode_action_limit'] + 1):
            if time.monotonic() >= deadline:
                raise TimeoutError('Periodic held-out evaluation exceeded its allowance')
            mask = action_mask(env) if metadata['workload']['action_masking'] else None
            action = greedy_action(torch, model, stack, mask=mask)
            frame, reward, done, info = env.step(action)
            histogram[action] += 1
            illegal += int(info['illegal_action'])
            rewards += float(reward)
            stack = shift_frame_stack(stack, frame)
            dead = info['inventory']['health'] <= 0
            success = goal_achieved(info['achievements']) and not dead
            if success or done:
                break
        rows.append({'seed': seed, 'steps': step, 'success': success, 'death': dead,
                     'illegal_actions': illegal, 'action_histogram': histogram,
                     'native_return': rewards,
                     'achievements': {k: v for k, v in info['achievements'].items() if v}})
    result = {'checkpoint': str(checkpoint), 'checkpoint_sha256': digest,
              'training_updates': metadata['training_steps'], 'workload': metadata['workload'],
              'environment': metadata['environment'],
              'episodes': rows, 'successes': sum(row['success'] for row in rows),
              'elapsed_seconds': time.monotonic() - started,
              'device': 'cpu', 'training_replay_connected': False}
    output.parent.mkdir(exist_ok=True, parents=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.output)
    print(json.dumps({key: report[key] for key in ('training_updates', 'successes', 'elapsed_seconds')}))
