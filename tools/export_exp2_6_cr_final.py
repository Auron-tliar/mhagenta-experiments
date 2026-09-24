"""Export completed CR runs with final policies and validation evidence only."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

import torch

from mha_exp_level2_cr.exp2_6.online_policy import SKILLS, tensor_hash
from mha_exp_level2_cr.exp2_6.online_runtime import MODULES


def export_final(source: Path, archive: Path) -> dict:
    """Create a new archive, preserving source files and omitting resumable state."""
    batch = json.loads((source / "batch.json").read_text())
    if batch["status"] != "completed" or len(batch["runs"]) != len(batch["run_ids"]):
        raise ValueError("Require a completed batch")
    with archive.open("xb") as output, tarfile.open(fileobj=output, mode="w:gz", compresslevel=1) as bundle:
        for row in batch["runs"]:
            run = source / f"run-{row['run']:05d}"
            config = json.loads((run / "config.json").read_text())
            result = json.loads((run / "result.json").read_text())
            if result != row["result"] or not result["execution_valid"] or result["errors"]:
                raise ValueError(f"Invalid result: {run}")
            agent = run / config["agent_id"] / "out"
            env = run / config["env_id"] / "out"
            learner = json.loads((agent / f"{config['agent_id']}.learner.json").read_text())
            files = [run / "config.json", run / "result.json",
                     run / f"{config['agent_id']}.log", run / f"{config['env_id']}.log",
                     env / f"{config['env_id']}.json", agent / "actions.jsonl", env / "actions.jsonl"]
            files.extend(agent / f"{config['agent_id']}.{name}.json" for name in MODULES)
            experience = sorted((agent / "experience").glob("segment-*.pt"))
            files.extend(experience)
            ll = json.loads((agent / f"{config['agent_id']}.ll_reasoner.json").read_text())
            if len(experience) != ll["segments"]:
                raise ValueError(f"Missing experience: {run}")
            for file in files:
                bundle.add(file, arcname=file.relative_to(source).as_posix(), recursive=False)
            for skill in SKILLS:
                saved = torch.load(agent / f"{skill}-trainer.pt", map_location="cpu", weights_only=False)
                if saved["summary"] != learner["skills"][skill] or tensor_hash(saved["model"]) != saved["summary"]["sha256"]:
                    raise ValueError(f"Final policy mismatch: {run}, {skill}")
                policy = io.BytesIO()
                torch.save({"model": saved["model"], "summary": saved["summary"]}, policy)
                member = tarfile.TarInfo((agent / f"{skill}-policy.pt").relative_to(source).as_posix())
                member.size = policy.tell()
                member.mode = 0o644
                policy.seek(0)
                bundle.addfile(member, policy)
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"archive": str(archive), "bytes": archive.stat().st_size, "sha256": digest, "run_ids": batch["run_ids"]}


def main() -> None:
    """Export one completed shard to a fresh transport archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(export_final(args.source.resolve(), args.archive.resolve())))


if __name__ == "__main__":
    main()
