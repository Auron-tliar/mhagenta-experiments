"""Explicit qualified policy families for matched2-5-BW; historical defaults stay frozen."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .policy import POLICY_ARCHITECTURE, build_q_network, file_sha256


def artifact_root() -> Path:
    """Locate packaged immutable policy families in both host and agent runtimes."""
    return Path(__file__).with_name('artifacts')


def resolve(family: str, *, table_len: int, num_blocks: int) -> tuple[Path, dict[str, Any]]:
    """Verify the complete family lineage and selected size's qualification and hashes."""
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]+', family):
        raise ValueError('Invalid artifact family name.')
    root = artifact_root() / family
    manifest_path = root / 'family.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 'transfer-family-v1' or manifest.get('family') != family:
        raise ValueError('Policy family manifest identity mismatch.')
    entries = manifest['policies']
    if set(entries) != {'5x8', '4x6', '7x12'}:
        raise ValueError('Policy family must contain all three qualified sizes.')
    for size, entry in entries.items():
        path = root / size / 'transfer-policy.pt'
        qualification_path = path.with_name('qualification.json')
        qualification = json.loads(qualification_path.read_text(encoding='utf-8'))
        columns, blocks = map(int, size.split('x'))
        assessment = qualification['assessment']
        audit_path = path.with_name('cpu-audit.json')
        assessment_path = path.with_name('assessment.json')
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
        if (file_sha256(path) != entry['checkpoint_sha256']
                or file_sha256(qualification_path) != entry['qualification_sha256']
                or qualification['checkpoint_sha256'] != entry['checkpoint_sha256']
                or not qualification['accepted'] or not qualification['cpu_audit_valid']
                or not qualification['development_passed']
                or (qualification['table_len'], qualification['num_blocks']) != (columns, blocks)
                or assessment['singles'] != 1000 or assessment['sequences'] != 100
                or assessment['requested_transfers'] != 2000
                or assessment['single_successes'] < 990 or assessment['sequence_successes'] < 95
                or assessment['successful_transfers'] < 1980):
            raise ValueError(f'Unqualified or changed family artifact: {size}')
        if (file_sha256(audit_path) != qualification['cpu_audit_sha256']
                or file_sha256(assessment_path) != qualification['assessment_sha256']
                or audit['checkpoint_sha256'] != entry['checkpoint_sha256']
                or not audit['valid'] or not audit['greedy_actions_reproduced']
                or not audit['numeric_symbolic_states_agree_after_every_action']
                or any(audit[key] != value for key, value in assessment.items())):
            raise ValueError('Family CPU assessment evidence mismatch.')
        parent = None if size == '5x8' else entries['5x8']['checkpoint_sha256']
        initialization = 'random' if size == '5x8' else 'semantic-resize'
        if (qualification['parent_sha256'] != parent or qualification['initialization'] != initialization
                or entry['parent_sha256'] != parent
                or entry['selected_optimizer_steps'] != qualification['selected_optimizer_steps']):
            raise ValueError('Family parent lineage mismatch.')
    size = f'{table_len}x{num_blocks}'
    if size not in entries:
        raise ValueError('No qualified family policy for this size.')
    entry = entries[size]
    return root / size / 'transfer-policy.pt', {
        'kind': 'family', 'family': family, 'table_len': table_len, 'num_blocks': num_blocks,
        'checkpoint_sha256': entry['checkpoint_sha256'],
        'manifest_sha256': entry['qualification_sha256'], 'family_sha256': file_sha256(manifest_path),
    }


def load(torch: Any, family: str, *, table_len: int, num_blocks: int) -> tuple[Any, dict]:
    """Load and freeze native-size CPU weights only after verifying the family contract."""
    path, reference = resolve(family, table_len=table_len, num_blocks=num_blocks)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    qualification = json.loads(path.with_name('qualification.json').read_text(encoding='utf-8'))
    if (payload['format_version'] != 'transfer-family-v1' or payload['family'] != family
            or payload['architecture'] != POLICY_ARCHITECTURE
            or (payload['table_len'], payload['num_blocks']) != (table_len, num_blocks)):
        raise ValueError('Family checkpoint metadata mismatch.')
    if (payload['provenance']['parent_sha256'] != qualification['parent_sha256']
            or payload['provenance']['initialization'] != qualification['initialization']
            or payload['optimizer_steps'] != qualification['selected_optimizer_steps']):
        raise ValueError('Checkpoint lineage differs from its qualification.')
    model = build_q_network(torch, table_len=table_len, num_blocks=num_blocks)
    model.load_state_dict(payload['model_state_dict'], strict=True)
    for parameter in model.parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise ValueError('Non-finite family weights.')
        parameter.requires_grad_(False)
    return model.eval(), reference
