"""Freeze a development-selected policy and assess unseen cases with CPU action audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from audit_resized import symbolic_facts
from family_support import CAMPAIGN, SIZES, development, freeze_protocol, load_model, source_hashes, tensor_digest, verify_baselines
from train_resized import evaluate, write_json
from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import TransferSpec
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_5.policy import file_sha256, greedy_inference


def passed(assessment: dict) -> bool:
    """Apply every declared aggregate gate with the full requested denominator."""
    return (assessment['singles'] == 1000 and assessment['sequences'] == 100
            and assessment['requested_transfers'] == 2000
            and assessment['single_successes'] >= 990
            and assessment['sequence_successes'] >= 95
            and assessment['successful_transfers'] >= 1980
            and all(not item['illegal_action'] for row in assessment['records'] for item in row['results']))


def audit(torch: object, model: object, dimensions: dict, assessment: dict, seed_start: int) -> dict:
    """Replay GPU evidence with CPU decisions and an independent symbolic environment."""
    numeric = BlocksWorldEnv(**dimensions, symbolic=False)
    symbolic = BlocksWorldEnv(**dimensions, symbolic=True)
    numeric.expose_snapshot = symbolic.expose_snapshot = True
    actions = successes = singles = sequences = executed = 0
    before = tensor_digest(model)
    try:
        assert len(assessment['records']) == 1100
        for index, row in enumerate(assessment['records']):
            seed, requested = seed_start + index, 1 if index < 1000 else 10
            assert row['seed'] == seed and row['requested'] == requested
            observation, _ = numeric.reset(seed=seed)
            symbols, _ = symbolic.reset(seed=seed)
            row_successes = 0
            assert 1 <= len(row['results']) <= requested
            for transfer_index, transfer in enumerate(row['results']):
                facts = symbolic_facts(symbols)
                assert ground_observation(observation, **dimensions).facts == facts
                candidates = enumerate_transfer_targets(facts)
                spec = TransferSpec.from_mapping(transfer['goal'])
                assert spec == candidates[(seed + transfer_index) % len(candidates)]
                assert 1 <= len(transfer['actions']) <= 32
                assert transfer['episode_length'] == len(transfer['actions'])
                for step, action in enumerate(transfer['actions']):
                    selected, _, _ = greedy_inference(torch, model, observation, spec, **dimensions)
                    if selected != action:
                        raise ValueError(f'GPU/CPU action mismatch: seed={seed}, transfer={transfer_index}, step={step}, GPU={action}, CPU={selected}')
                    observation, _, _, _, numeric_info = numeric.step(action)
                    symbols, _, _, _, symbolic_info = symbolic.step(action)
                    assert numeric_info['snapshot'].legal and symbolic_info['snapshot'].legal
                    facts = symbolic_facts(symbols)
                    assert ground_observation(observation, **dimensions).facts == facts
                    succeeded = (spec.target_facts | {'hand-empty()'}).issubset(facts)
                    if succeeded:
                        assert step == len(transfer['actions']) - 1
                    actions += 1
                assert succeeded == transfer['success'] and not transfer['illegal_action']
                assert transfer['outcome'] == ('succeeded' if succeeded else 'step-limit')
                if not succeeded:
                    assert len(transfer['actions']) == 32
                    assert transfer_index == len(row['results']) - 1
                successes += succeeded
                row_successes += succeeded
                executed += 1
            complete = row_successes == requested
            assert complete == row['success']
            if index < 1000:
                singles += complete
            else:
                sequences += complete
    finally:
        numeric.close()
        symbolic.close()
    expected = {'single_successes': singles, 'singles': 1000, 'sequence_successes': sequences,
                'sequences': 100, 'successful_transfers': successes, 'requested_transfers': 2000,
                'executed_transfers': executed, 'actions': actions}
    assert all(assessment[key] == value for key, value in expected.items())
    assert tensor_digest(model) == before
    return {'valid': True, 'device': 'cpu', 'torch_threads': 1,
            'numeric_symbolic_states_agree_after_every_action': True,
            'greedy_actions_reproduced': True, 'tensor_sha256': before, **expected}


def qualify(directory: Path) -> dict:
    """Retain development, candidate freeze, final cohort and independent CPU replay."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seeds = freeze_protocol()
    verify_baselines(seeds)
    training = json.loads((directory / 'report.json').read_text(encoding='utf-8'))
    if training['preflight'] or training['status'] != 'completed':
        raise ValueError('Qualification requires a completed full training attempt.')
    output = directory / 'qualification'
    output.mkdir(exist_ok=False)
    candidate = directory / 'best.pt'
    if file_sha256(candidate) != training['selected_sha256']:
        raise ValueError('Selected checkpoint changed after training.')
    model, payload = load_model(torch, candidate, 'cuda')
    size = training['size']
    dimensions = dict(zip(('table_len', 'num_blocks'), SIZES[size]))
    expanded = development(torch, model, dimensions)
    write_json(output / 'development-selection.json', expanded)
    revealed = evaluate(torch, model, dimensions, seed_start=10_000_000, singles=1000, sequences=100)
    write_json(output / 'development-revealed.json', revealed)
    regressions = expanded['regressions']
    development_ok = passed(revealed) and regressions['passed'] == regressions['total']
    if size == '5x8':
        legacy_single = evaluate(torch, model, dimensions, seed_start=370_000, singles=100, sequences=0)
        legacy_sequence = evaluate(torch, model, dimensions, seed_start=371_000, singles=0, sequences=20)
        write_json(output / 'development-legacy.json', {'single': legacy_single, 'sequence': legacy_sequence})
        development_ok &= (legacy_single['successful_transfers'] == 100
                           and legacy_sequence['successful_transfers'] == 200)
    if not development_ok:
        report = {'accepted': False, 'stage': 'development', 'final_assessment_performed': False,
                  'checkpoint_sha256': training['selected_sha256']}
        write_json(output / 'qualification.json', report)
        return report
    start = seeds['allocations'][size]['final'][0]
    # A consumed cohort is never reused automatically after it informs a refinement.
    reservation = CAMPAIGN / f'final-{start}-consumed.json'
    with reservation.open('x', encoding='utf-8') as stream:
        json.dump({'size': size, 'checkpoint_sha256': training['selected_sha256'],
                   'attempt': str(directory), 'frozen_unix': time.time()}, stream, indent=2)
    shutil.copyfile(candidate, output / 'transfer-policy.pt')
    checkpoint_hash = file_sha256(output / 'transfer-policy.pt')
    assert checkpoint_hash == training['selected_sha256']
    frozen = {'checkpoint_sha256': checkpoint_hash, 'selected_optimizer_steps': payload['optimizer_steps'],
              'seed_start': start, 'singles': 1000, 'sequences': 100, 'sequence_length': 10,
              'inference': 'greedy legal masked,32actions', 'sources': source_hashes(),
              'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name(0),
              'seed_manifest_sha256': file_sha256(CAMPAIGN / 'seed-manifest.json')}
    write_json(output / 'candidate-freeze.json', frozen)
    print(f'Candidate frozen for{size}; assessing unseen cohort{start}.', flush=True)
    assessment = evaluate(torch, model, dimensions, seed_start=start, singles=1000, sequences=100)
    write_json(output / 'assessment.json', assessment)
    cpu, _ = load_model(torch, output / 'transfer-policy.pt', 'cpu')
    try:
        verification = audit(torch, cpu, dimensions, assessment, start)
    except Exception as exc:
        write_json(output / 'cpu-audit-failure.json', {'error': repr(exc), 'checkpoint_sha256': checkpoint_hash})
        raise
    verification.update(torch=str(torch.__version__), checkpoint_sha256=checkpoint_hash)
    write_json(output / 'cpu-audit.json', verification)
    assert file_sha256(output / 'transfer-policy.pt') == checkpoint_hash
    verify_baselines(seeds)
    report = {'accepted': passed(assessment), 'stage': 'final', 'size': size, **dimensions,
              'checkpoint_sha256': checkpoint_hash, 'parent_sha256': payload['provenance']['parent_sha256'],
              'initialization': payload['provenance']['initialization'],
              'selected_optimizer_steps': payload['optimizer_steps'],
              'training_optimizer_steps': training['optimizer_steps'],
              'assessment': {key: value for key, value in assessment.items() if key != 'records'},
              'cpu_audit_valid': verification['valid'], 'development_passed': True,
              'final_seed_start': start, 'final_assessment_performed': True,
              'training_report_sha256': file_sha256(directory / 'report.json'),
              'assessment_sha256': file_sha256(output / 'assessment.json'),
              'cpu_audit_sha256': file_sha256(output / 'cpu-audit.json')}
    write_json(output / 'qualification.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(qualify(args.directory.resolve()), indent=2), flush=True)
