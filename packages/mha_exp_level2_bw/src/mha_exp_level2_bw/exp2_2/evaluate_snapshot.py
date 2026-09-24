"""Independent CPU evaluation of an immutable periodic BW checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import time


def evaluate(checkpoint: Path, output: Path, *, timeout: float = 300.0) -> dict:
    """Use the final-evaluation seeds and goal sampler without training replay."""
    import numpy as np
    import torch
    from mha_env_blocksworld import BlocksWorldEnv
    from .policy import TABLE_LEN, NUM_BLOCKS, load_policy_checkpoint, sample_goal, goal_achieved
    from .play_policy import greedy_action
    from .protocol import DQNProtocol

    torch.set_num_threads(1)
    started = time.monotonic()
    model, metadata = load_policy_checkpoint(torch, checkpoint)
    protocol = DQNProtocol.from_record(metadata['training_protocol'])
    if not metadata['frozen_for_evaluation']:
        raise ValueError('Periodic evaluation requires an immutable frozen snapshot')
    rows = []
    for seed in protocol.evaluation_seeds:
        env = BlocksWorldEnv(table_len=TABLE_LEN, num_blocks=NUM_BLOCKS, symbolic=False)
        observation, _ = env.reset(seed=seed)
        goal = sample_goal(np.random.default_rng(seed), observation)
        illegal = 0
        histogram = [0] * 4
        success = False
        try:
            for step in range(1, protocol.max_episode_length + 1):
                if time.monotonic() - started >= timeout:
                    raise TimeoutError('BW periodic evaluation exceeded its allowance')
                action = greedy_action(torch, model, observation, goal, action_masking=protocol.action_masking)
                observation, _, _, _, _ = env.step(action)
                illegal += int(not env._last_legal)
                histogram[action] += 1
                success = goal_achieved(observation, goal)
                if success:
                    break
            rows.append(dict(seed=seed, goal=list(goal), success=success, steps=step,
                             illegal_actions=illegal, action_histogram=histogram))
        finally:
            env.close()
    result = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                  training_updates=metadata['training_steps'], protocol=protocol.record(), episodes=rows,
                  successes=sum(row['success'] for row in rows), elapsed_seconds=time.monotonic()-started,
                  device='cpu', training_replay_connected=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(output)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.checkpoint, args.output)
    print(json.dumps({k: report[k] for k in ('training_updates', 'successes', 'elapsed_seconds')}))
