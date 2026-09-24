"""Seed isolation, lineage and development checks for the September policy family."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import TransferSpec
from mha_exp_level2_bw.exp2_5.evaluation import evaluate_transfer
from mha_exp_level2_bw.exp2_5.grounding import ground_observation
from mha_exp_level2_bw.exp2_5.policy import build_q_network, file_sha256
from mha_exp_level2_bw.exp2_5.resized import feature_keys
from regression import PLANNER_REGRESSIONS
from train_resized import evaluate, write_json

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / 'results/2-5-bw-policy-family-20260920'
BASELINE = ROOT / 'results/2-5-bw-5x8-extended-assessment-20260920-1'
ARTIFACTS = ROOT / 'packages/mha_exp_level2_bw/src/mha_exp_level2_bw/exp2_5/artifacts'
FAMILY = 'structured-dqfd-family-20260920-v1'
SIZES = {'5x8': (5, 8), '4x6': (4, 6), '7x12': (7, 12)}


def tensor_digest(model: Any) -> str:
    """Identify the initial or selected tensors independently of checkpoint metadata."""
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(f'{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def source_hashes() -> dict[str, str]:
    """Bind preparation, deployment primitives and native environment sources."""
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += list((ARTIFACTS.parent).glob('*.py'))
    paths += list((ROOT / 'packages/mha_env_blocksworld/src').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(paths)}


def freeze_protocol() -> dict[str, Any]:
    """Create the seed allocation once, before any new-family learning or selection."""
    path = CAMPAIGN / 'seed-manifest.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    CAMPAIGN.mkdir(parents=True, exist_ok=False)
    allocations = {}
    for index, (size, dimensions) in enumerate(SIZES.items()):
        start = (11 + index) * 1_000_000
        allocations[size] = {
            'table_len': dimensions[0], 'num_blocks': dimensions[1],
            'training': [7_000_000, 7_499_999], 'demonstrations': [8_000_000, 8_099_999],
            'selection': [9_000_000, 9_000_599],
            'revealed_development': [10_000_000, 10_001_099],
            'final': [start, start + 1099], 'final_singles': [start, start + 999],
            'final_sequences': [start + 1000, start + 1099],
        }
    protocol = {
        'family': FAMILY, 'case_identity': ['table_len', 'num_blocks', 'seed'],
        'allocations': allocations, 'excluded_main_seeds': [23000, 23049],
        'reserved_in_every_size': [[11_000_000, 11_001_099], [12_000_000, 12_001_099],
                                   [13_000_000, 13_001_099]],
        'initialization_seeds': {'5x8': 202609205, '4x6': 202609204, '7x12': 202609207},
        'old_regressions': 'Seven revealed failures reconstructed with saved prefixes; legacy seed1000 regression and 370000/371000 certification used only as development.',
        'demonstration_rule': 'Expert actions on evolving worlds; no main or final cases, no revealed regression states used as demonstrations.',
        'selection_rule': 'Lexicographic exact-regression passes, successful requested transfers, complete sequences, negative action count; ties retain earlier weights.',
        'selection_suite': {'singles': 500, 'sequences': 100, 'sequence_length': 10},
        'early_stop': 'Three consecutive perfect expanded selection suites and all exact regressions, after20000 scratch or6000 child environment actions; final gate still required.',
        'final_gate': {'single_successes': 990, 'sequence_successes': 95, 'successful_transfers': 1980,
                       'singles': 1000, 'sequences': 100, 'requested_transfers': 2000},
        'reuse_rule': 'A failed final cohort influencing refinement is retired into development; a new disjoint block must be declared before the next assessment.',
        'main_recovery': 'Preserve completed valid runs; repair technical faults and resume only interrupted/failed and remaining IDs. Retain scientific failures.',
        'baseline_hashes': {p.relative_to(ROOT).as_posix(): file_sha256(p)
                            for p in sorted(ARTIFACTS.rglob('*')) if p.is_file()},
        'revealed_evidence_hashes': {p.name: file_sha256(p) for p in BASELINE.iterdir() if p.is_file()},
    }
    write_json(path, protocol)
    return protocol


def verify_baselines(protocol: dict[str, Any]) -> None:
    """Refuse to continue if historical policy or assessment evidence changed."""
    for relative, expected in protocol['baseline_hashes'].items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f'Historical artifact changed: {relative}')
    for name, expected in protocol['revealed_evidence_hashes'].items():
        if file_sha256(BASELINE / name) != expected:
            raise ValueError(f'Revealed evidence changed: {name}')


def load_model(torch: Any, path: Path, device: str = 'cpu') -> tuple[Any, dict]:
    """Load a new-family checkpoint with native dimensions and finite weights."""
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format_version'] != 'transfer-family-v1':
        raise ValueError('Expected a new-family checkpoint, not historical weights.')
    model = build_q_network(torch, table_len=payload['table_len'], num_blocks=payload['num_blocks'])
    model.load_state_dict(payload['model_state_dict'], strict=True)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('Non-finite model weights.')
    return model.to(device).eval(), payload


def initialize(torch: Any, size: str, parent: Path | None) -> tuple[Any, dict]:
    """Initialize random5/8 or semantically resize an explicitly qualified new parent."""
    columns, blocks = SIZES[size]
    model = build_q_network(torch, table_len=columns, num_blocks=blocks)
    if size == '5x8':
        if parent is not None:
            raise ValueError('Scratch initialization cannot accept a parent.')
        return model, {'initialization': 'random', 'parent_sha256': None,
                       'initial_tensor_sha256': tensor_digest(model)}
    if parent is None:
        raise ValueError('A qualified new5/8 parent must be explicit.')
    qualification = json.loads((parent.parent / 'qualification.json').read_text(encoding='utf-8'))
    digest = file_sha256(parent)
    measured = qualification['assessment']
    if (not qualification['accepted'] or qualification['checkpoint_sha256'] != digest
            or not qualification['cpu_audit_valid'] or not qualification['development_passed']
            or measured['single_successes'] < 990 or measured['sequence_successes'] < 95
            or measured['successful_transfers'] < 1980 or measured['requested_transfers'] != 2000
            or file_sha256(parent.parent / 'cpu-audit.json') != qualification['cpu_audit_sha256']
            or file_sha256(parent.parent / 'assessment.json') != qualification['assessment_sha256']):
        raise ValueError('Parent has not passed qualification.')
    original, payload = load_model(torch, parent)
    if (payload['table_len'], payload['num_blocks']) != (5, 8) or payload['provenance']['initialization'] != 'random':
        raise ValueError('Children must independently descend from the new random5/8.')
    first = 'entity_encoder.0.weight'
    state, source = model.state_dict(), original.state_dict()
    keys = {key: index for index, key in enumerate(feature_keys(5, 8))}
    mapping = [(i, keys[key]) for i, key in enumerate(feature_keys(columns, blocks)) if key in keys]
    for name in state:
        if name != first:
            state[name].copy_(source[name])
    state[first].zero_()
    for destination, origin in mapping:
        state[first][:, destination].copy_(source[first][:, origin])
    model.load_state_dict(state)
    return model, {'initialization': 'semantic-resize', 'parent_sha256': digest,
                   'parent_selected_updates': payload['optimizer_steps'],
                   'input_mapping': mapping, 'source_features': feature_keys(5, 8),
                   'destination_features': feature_keys(columns, blocks), 'new_connections': 'zero',
                   'stack_alignment': 'bottom', 'column_alignment': 'left',
                   'initial_tensor_sha256': tensor_digest(model), 'all_weights_trainable': True}


def regression_cases() -> list[dict]:
    """Recover the exact seven failed start states plus the original planner regression."""
    assessment = json.loads((BASELINE / 'assessment.json').read_text(encoding='utf-8'))
    diagnostics = json.loads((BASELINE / 'failure-diagnostics.json').read_text(encoding='utf-8'))
    expected = {row['seed']: row for row in diagnostics['failures']}
    cases = []
    for row in assessment['records']:
        if row['success']:
            continue
        prefix = [a for transfer in row['results'][:-1] for a in transfer['actions']]
        cases.append({'seed': row['seed'], 'prefix': prefix, 'goal': row['results'][-1]['goal'],
                      'facts': expected[row['seed']]['recorded_states'][0]['facts']})
    if len(cases) != 7:
        raise ValueError('Expected the seven retained assessment failures.')
    cases += [{'seed': case.environment_seed, 'prefix': list(case.action_prefix),
               'goal': case.spec.as_dict()} for case in PLANNER_REGRESSIONS]
    return cases


def regressions(torch: Any, model: Any) -> dict:
    """Evaluate new weights from original failed states, without changing their prefixes."""
    environment = BlocksWorldEnv(table_len=5, num_blocks=8, symbolic=False)
    environment.expose_snapshot = True
    records = []
    try:
        for case in regression_cases():
            observation, _ = environment.reset(seed=case['seed'])
            for action in case['prefix']:
                observation, _, _, _, info = environment.step(action)
                if not info['snapshot'].legal:
                    raise ValueError('Illegal saved regression prefix.')
            facts = sorted(ground_observation(observation).facts)
            if 'facts' in case and facts != case['facts']:
                raise ValueError('Reconstructed failed state differs from retained diagnostics.')
            result, _ = evaluate_transfer(torch, model, environment, observation,
                                          TransferSpec.from_mapping(case['goal']), case['seed'])
            records.append({**case, 'start_facts': facts, 'result': asdict(result)})
    finally:
        environment.close()
    return {'passed': sum(row['result']['success'] for row in records), 'total': len(records), 'records': records}


def development(torch: Any, model: Any, dimensions: dict, *, preflight: bool = False) -> dict:
    """Run expanded selection and exact-state regressions using development seeds only."""
    evaluation = evaluate(torch, model, dimensions, seed_start=9_000_000,
                          singles=3 if preflight else 500, sequences=1 if preflight else 100)
    if dimensions == {'table_len': 5, 'num_blocks': 8}:
        evaluation['regressions'] = regressions(torch, model)
    else:
        evaluation['regressions'] = {'passed': 0, 'total': 0, 'records': []}
    return evaluation
