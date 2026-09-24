"""Train one local-CUDA stage of the explicitly versioned Transfer policy family."""

from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback

import numpy as np

from evaluation import transition_reward
from family_support import CAMPAIGN, FAMILY, ROOT, SIZES, development, freeze_protocol, initialize, load_model, regression_cases, source_hashes, tensor_digest, verify_baselines
from selected_dqfd import CONFIGS, VARIANT_DQFD_LITE, RawTransition, SelectedReplayBuffer, _optimize, emit_n_step
from train_resized import collect_demonstrations, expert_action, save_checkpoint, write_json
from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation, transfer_phase
from mha_exp_level2_bw.exp2_5.policy import MAX_POLICY_STEPS, POLICY_ARCHITECTURE, condition_observation, file_sha256, greedy_inference, legal_action_indices
from mha_exp_level2_bw.exp2_5.contracts import TransferSpec


def add_regression_demonstrations(replay: SelectedReplayBuffer, repetitions: int) -> dict:
    """Label expert corrections from revealed exact states as protected training data."""
    environment = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    environment.expose_snapshot = True
    steps = 0
    try:
        for _ in range(repetitions):
            for case in regression_cases():
                observation, _ = environment.reset(seed=case['seed'])
                for action in case['prefix']:
                    observation, _, _, _, info = environment.step(action)
                    if not info['snapshot'].legal:
                        raise ValueError('Illegal regression reconstruction action.')
                grounded = ground_observation(observation)
                spec = TransferSpec.from_mapping(case['goal'])
                phase = transfer_phase(grounded, spec)
                pending = deque()
                for _ in range(32):
                    state = condition_observation(observation, spec)
                    action = expert_action(grounded, spec)
                    observation, _, _, _, info = environment.step(action)
                    if not info['snapshot'].legal:
                        raise ValueError('Illegal corrective demonstration action.')
                    grounded = ground_observation(observation)
                    reward, outcome, phase = transition_reward(grounded, spec, phase, False)
                    terminal = outcome is not None
                    pending.append(RawTransition(state, action, reward, condition_observation(observation, spec), terminal, demonstration=True))
                    for transition in emit_n_step(pending, 3, flush=terminal):
                        replay.append(transition)
                    steps += 1
                    if terminal:
                        if outcome != 'succeeded':
                            raise ValueError('Corrective expert did not succeed.')
                        break
                else:
                    raise ValueError('Corrective expert exceeded32actions.')
    finally:
        environment.close()
    return {'steps': steps, 'episodes': repetitions * 8, 'repetitions': repetitions,
            'cases': regression_cases(), 'role': 'training; revealed regressions are not held-out evidence'}


def train(args: argparse.Namespace) -> dict:
    """Train with protected DQfD demonstrations and evolving worlds; never touch final cases."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('Local CUDA is required; no implicit CPU training fallback.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = freeze_protocol()
    verify_baselines(seeds)
    seed = args.seed or seeds['initialization_seeds'][args.size]
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(seed)
    scratch = args.size == '5x8'
    columns, blocks = SIZES[args.size]
    dimensions = {'table_len': columns, 'num_blocks': blocks}
    steps = args.steps or (250_000 if scratch else 100_000)
    learning_rate = 1e-4 if scratch else 5e-5
    epsilon_start, epsilon_end, epsilon_decay = (1.0, 0.05, 100_000) if scratch else (0.1, 0.02, 20_000)
    demonstrations = 128 if args.preflight else args.demonstrations
    warm_updates = 8 if args.preflight else 500
    if args.continue_from is not None:
        learning_rate = 5e-5
        epsilon_start, epsilon_end, epsilon_decay = 0.1, 0.02, 20_000
    interval = 2000
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    progress = {'status': 'initializing', 'environment_steps': 0, 'optimizer_steps': 0}

    def publish(status: str, **values: object) -> None:
        progress.update(values, status=status, elapsed_seconds=time.monotonic() - started, updated_unix=time.time())
        write_json(output / 'progress.json', progress)
        print(json.dumps(progress, allow_nan=False), flush=True)

    try:
        if args.continue_from is None:
            model, provenance = initialize(torch, args.size, args.parent)
        else:
            if args.parent is not None or args.size != '5x8':
                raise ValueError('This refinement continuation supports only the new scratch5/8 lineage.')
            model, prior = load_model(torch, args.continue_from)
            if prior['family'] != FAMILY or prior['provenance']['initialization'] != 'random':
                raise ValueError('Continuation must descend from this new random family.')
            provenance = {**prior['provenance'], 'continuation_sha256': file_sha256(args.continue_from),
                          'continuation_from_optimizer_steps': prior['optimizer_steps'],
                          'continuation_start_tensor_sha256': tensor_digest(model)}
            progress['optimizer_steps'] = prior['optimizer_steps']
        initial_updates = progress['optimizer_steps']
        model.to('cuda').train()
        target = deepcopy(model).eval()
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        metadata = {'format_version': 'transfer-family-v1', 'family': FAMILY,
                    'architecture': POLICY_ARCHITECTURE, **dimensions, 'n_actions': 4,
                    'observation_shape': [blocks + 2, columns, blocks],
                    'model_input_shape': [blocks + 4, columns, blocks], 'provenance': provenance}
        protocol = {**metadata, 'seed': seed, 'torch': str(torch.__version__),
                    'python': platform.python_version(), 'device': 'cuda', 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(0), 'torch_threads': 1,
                    'algorithm': VARIANT_DQFD_LITE, 'max_environment_steps': steps,
                    'demonstration_steps_minimum': demonstrations, 'warm_updates': warm_updates,
                    'learning_rate': learning_rate, 'weight_decay': 0.01, 'batch_size': 128,
                    'discount': 0.99, 'gradient_clip': 10, 'target_sync_updates': 500,
                    'replay_capacity': 50_000 + demonstrations + 32 + args.regression_demos * 8 * 32, 'priority_alpha': 0.6,
                    'priority_beta': [0.4, 1.0], 'priority_epsilon': 1e-5,
                    'demonstration_priority_bonus': 0.1, 'margin': 0.8, 'margin_weight': 1.0,
                    'n_step': 3, 'epsilon_start': epsilon_start, 'epsilon_end': epsilon_end,
                    'epsilon_decay_steps': epsilon_decay, 'selection_interval': interval,
                    'seed_manifest_sha256': file_sha256(CAMPAIGN / 'seed-manifest.json'),
                    'sources': source_hashes(), 'preflight': args.preflight,
                    'allocation': seeds['allocations'][args.size],
                    'selection_rule': seeds['selection_rule'], 'early_stop': seeds['early_stop'],
                    'legal_masked_training': True, 'max_actions_per_transfer': 32,
                    'corrective_regression_demo_repetitions': args.regression_demos,
                    'continuation': str(args.continue_from) if args.continue_from is not None else None,
                    'world_reset': 'At ten successful transfers or the first failure; no main task seeds.',
                    'initialization_optimizer_replay': 'Fresh optimizer/replay; target copied from initialized online model.'}
        write_json(output / 'protocol.json', protocol)
        for relative, expected in protocol['sources'].items():
            destination = output / 'executed-sources' / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
            if file_sha256(destination) != expected:
                raise ValueError('Preparation source changed during snapshot capture.')
        save_checkpoint(torch, output / 'initial.pt', model, {**metadata, 'optimizer_steps': initial_updates, 'environment_steps': 0})
        history, best_score, best_update, perfect_streak = [], None, 0, 0

        def select() -> bool:
            nonlocal best_score, best_update, perfect_streak
            publish('selection')
            evaluation = development(torch, model, dimensions, preflight=args.preflight)
            summary = {k: v for k, v in evaluation.items() if k not in {'records', 'regressions'}}
            regression = evaluation['regressions']
            summary.update(regressions_passed=regression['passed'], regressions_total=regression['total'])
            updates = progress['optimizer_steps']
            write_json(output / f'selection-{updates:07d}.json', evaluation)
            score = (regression['passed'], summary['successful_transfers'], summary['sequence_successes'], -summary['actions'])
            if best_score is None or score > best_score:
                best_score, best_update = score, updates
                save_checkpoint(torch, output / 'best.pt', model, {**metadata, 'optimizer_steps': updates,
                                'environment_steps': progress['environment_steps'], 'selection': summary})
            save_checkpoint(torch, output / 'latest.pt', model, {**metadata, 'optimizer_steps': updates,
                            'environment_steps': progress['environment_steps']})
            perfect = (summary['successful_transfers'] == summary['requested_transfers']
                       and regression['passed'] == regression['total'])
            perfect_streak = perfect_streak + 1 if perfect else 0
            history.append({'optimizer_steps': updates, 'environment_steps': progress['environment_steps'], **summary})
            write_json(output / 'selection-history.json', history)
            publish('training', selection=summary, best_optimizer_steps=best_update)
            return perfect_streak >= 3 and progress['environment_steps'] >= (20_000 if scratch else 6000)

        select()
        protected_capacity = demonstrations + 32 + args.regression_demos * 8 * 32
        replay = SelectedReplayBuffer(50_000 + protected_capacity, prioritized=True,
                                      protected_capacity=protected_capacity,
                                      input_shape=(blocks + 4, columns, blocks))
        publish('demonstrations')
        demos = collect_demonstrations(replay, dimensions, rng, demonstrations)
        if args.regression_demos:
            corrections = add_regression_demonstrations(replay, args.regression_demos)
            write_json(output / 'corrective-demonstrations.json', corrections)
            demos['diverse_steps'] = demos['steps']
            demos['steps'] += corrections['steps']
            demos['episodes'] += corrections['episodes']
        write_json(output / 'demonstrations.json', demos)
        replay.protected_capacity = len(replay)
        replay.next_index = len(replay)
        progress['demonstration_actions'] = demos['steps']

        def update() -> None:
            beta = 0.4 + 0.6 * min(1.0, progress['environment_steps'] / steps)
            loss, margin = _optimize(torch, model, target, optimizer, replay, rng, 'cuda',
                                     CONFIGS[VARIANT_DQFD_LITE], beta, mask_legal_actions=True)
            progress['optimizer_steps'] += 1
            count = progress['optimizer_steps']
            if count % 500 == 0:
                target.load_state_dict(model.state_dict())
            if count % 100 == 0:
                publish('training', loss=loss, margin_loss=margin)

        publish('warm-up')
        for _ in range(warm_updates):
            update()
        select()
        environment = BlocksWorldEnv(**dimensions, symbolic=False)
        environment.expose_snapshot = True
        pending = deque()
        grounded = None
        resets = episodes = successes = in_world = 0
        next_evaluation, stop = min(1000, steps), False
        try:
            while progress['environment_steps'] < steps and not stop:
                if grounded is None or in_world == 10:
                    if resets >= 500_000:
                        raise ValueError('Declared training seed block exhausted.')
                    observation, _ = environment.reset(seed=7_000_000 + resets)
                    resets += 1
                    in_world = 0
                    grounded = ground_observation(observation, **dimensions)
                candidates = enumerate_transfer_targets(grounded.facts)
                spec = candidates[int(rng.integers(len(candidates)))]
                phase = transfer_phase(grounded, spec)
                pending.clear()
                for episode_step in range(1, MAX_POLICY_STEPS + 1):
                    state = condition_observation(grounded.observation, spec, **dimensions)
                    epsilon = epsilon_start + (epsilon_end - epsilon_start) * min(1.0, progress['environment_steps'] / epsilon_decay)
                    if rng.random() < epsilon:
                        action = int(rng.choice(legal_action_indices(grounded.observation, **dimensions)))
                    else:
                        action, _, _ = greedy_inference(torch, model, grounded.observation, spec, **dimensions)
                    observation, _, _, _, info = environment.step(action)
                    if not info['snapshot'].legal:
                        raise RuntimeError('Legal-masked training executed an illegal action.')
                    grounded = ground_observation(observation, **dimensions)
                    reward, outcome, phase = transition_reward(grounded, spec, phase, False)
                    progress['environment_steps'] += 1
                    terminal = outcome is not None or episode_step == MAX_POLICY_STEPS or progress['environment_steps'] == steps
                    pending.append(RawTransition(state, action, reward, condition_observation(observation, spec, **dimensions), terminal))
                    for transition in emit_n_step(pending, 3, flush=terminal):
                        replay.append(transition)
                        update()
                    if terminal:
                        episodes += 1
                        in_world += 1
                        successes += outcome == 'succeeded'
                        progress.update(training_episodes=episodes, training_successes=successes, training_resets=resets)
                        if outcome != 'succeeded':
                            grounded = None
                        break
                if progress['environment_steps'] >= next_evaluation:
                    stop = select()
                    next_evaluation = min(steps, (progress['environment_steps'] // interval + 1) * interval)
        finally:
            environment.close()
        verify_baselines(seeds)
        report = {'status': 'completed', 'size': args.size, 'preflight': args.preflight,
                  'environment_steps': progress['environment_steps'], 'optimizer_steps': progress['optimizer_steps'],
                  'attempt_optimizer_steps': progress['optimizer_steps'] - initial_updates,
                  'demonstration_actions': demos['steps'], 'selected_optimizer_steps': best_update,
                  'selected_sha256': file_sha256(output / 'best.pt'), 'early_stopped': stop,
                  'elapsed_seconds': time.monotonic() - started, 'assessment_performed': False}
        write_json(output / 'report.json', report)
        publish('completed', report=report)
        return report
    except BaseException as exc:
        write_json(output / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(), **progress})
        publish('failed', error=repr(exc))
        raise


def main() -> None:
    """Train one ordered stage in a fresh output directory, without automatic qualification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--size', choices=SIZES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--demonstrations', type=int, default=8000)
    parser.add_argument('--regression-demos', type=int, default=0)
    parser.add_argument('--continue-from', type=Path)
    args = parser.parse_args()
    if args.steps is not None and args.steps <= 0:
        parser.error('The environment action budget must be positive.')
    if args.demonstrations < 8000 or args.regression_demos < 0 or (args.regression_demos and args.size != '5x8'):
        parser.error('At least8000 demonstrations; corrective regressions apply only to5/8.')
    train(args)


if __name__ == '__main__':
    main()
