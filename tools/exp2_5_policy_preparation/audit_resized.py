"""Independently replay a resized policy's assessment through symbolic Blocks World."""

import argparse
import json
from pathlib import Path

from mha_env_blocksworld import BlocksWorldEnv
from mha_exp_level2_bw.exp2_5.contracts import RUNTIME_TO_CANONICAL, TransferSpec, format_fact
from mha_exp_level2_bw.exp2_5.grounding import enumerate_transfer_targets, ground_observation
from mha_exp_level2_bw.exp2_5.policy import build_q_network, file_sha256, greedy_inference


def symbolic_facts(symbols: list[str]) -> set[str]:
    """Normalize the native symbolic environment's independent facts."""
    result = set()
    for text in symbols:
        name, _, arguments = text.partition("(")
        values = arguments.rstrip(")").split(",") if arguments else []
        result.add(format_fact(RUNTIME_TO_CANONICAL[name], [value.strip() for value in values]))
    return result


def audit(directory: Path) -> dict:
    """Verify checkpoint binding and every assessment action and success in both environments."""
    import torch

    torch.set_num_threads(1)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    assessment = json.loads((directory / "assessment.json").read_text())
    if file_sha256(directory / "transfer-policy.pt") != report["checkpoint_sha256"]:
        raise ValueError("Final checkpoint digest mismatch.")
    checkpoint = torch.load(directory / "transfer-policy.pt", map_location="cpu", weights_only=True)
    dimensions = {key: protocol[key] for key in ("table_len", "num_blocks")}
    if any(checkpoint[key] != value or report[key] != value for key, value in dimensions.items()):
        raise ValueError("World dimensions differ between saved artifacts.")
    model = build_q_network(torch, **dimensions).eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    numeric = BlocksWorldEnv(**dimensions, symbolic=False)
    symbolic = BlocksWorldEnv(**dimensions, symbolic=True)
    numeric.expose_snapshot = symbolic.expose_snapshot = True
    actions = successes = singles = sequences = executed = 0
    try:
        if len(assessment["records"]) != 1100:
            raise ValueError("Assessment does not contain the expected 1100 initial worlds.")
        for index, row in enumerate(assessment["records"]):
            seed = 10_000_000 + index
            requested = 1 if index < 1000 else 10
            assert row["seed"] == seed and row["requested"] == requested
            observation, _ = numeric.reset(seed=seed)
            symbols, _ = symbolic.reset(seed=seed)
            row_successes = 0
            assert 1 <= len(row["results"]) <= requested
            for transfer_index, transfer in enumerate(row["results"]):
                facts = symbolic_facts(symbols)
                assert ground_observation(observation, **dimensions).facts == facts
                candidates = enumerate_transfer_targets(facts)
                spec = TransferSpec.from_mapping(transfer["goal"])
                assert spec == candidates[(seed + transfer_index) % len(candidates)]
                assert 1 <= len(transfer["actions"]) <= 32
                assert transfer["episode_length"] == len(transfer["actions"])
                for step, action in enumerate(transfer["actions"]):
                    assert greedy_inference(torch, model, observation, spec, **dimensions)[0] == action
                    observation, _, _, _, numeric_info = numeric.step(action)
                    symbols, _, _, _, symbolic_info = symbolic.step(action)
                    assert numeric_info["snapshot"].legal and symbolic_info["snapshot"].legal
                    actions += 1
                    succeeded = (spec.target_facts | {"hand-empty()"}).issubset(symbolic_facts(symbols))
                    if succeeded:
                        assert step == len(transfer["actions"]) - 1
                assert succeeded == transfer["success"] and not transfer["illegal_action"]
                assert transfer["outcome"] == ("succeeded" if succeeded else "step-limit")
                if not succeeded:
                    assert len(transfer["actions"]) == 32
                    assert transfer_index == len(row["results"]) - 1
                successes += succeeded
                row_successes += succeeded
                executed += 1
            complete = row_successes == requested
            assert complete == row["success"]
            if index < 1000:
                singles += complete
            else:
                sequences += complete
    finally:
        numeric.close()
        symbolic.close()
    expected = {"single_successes": singles, "singles": 1000, "sequence_successes": sequences,
                "sequences": 100, "successful_transfers": successes, "requested_transfers": 2000,
                "executed_transfers": executed, "actions": actions}
    assert report["assessment"] == expected
    assert all(assessment[key] == value for key, value in expected.items())
    assert report["accepted"] == (singles >= 990 and sequences >= 95 and successes >= 1980)
    return {"valid": True, **dimensions, **expected, "checkpoint_sha256": report["checkpoint_sha256"]}


def main() -> None:
    """Read a completed result directory and print the audit, without modifying it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))


if __name__ == "__main__":
    main()
