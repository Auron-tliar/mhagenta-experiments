"""Persist compact EC2 heartbeats and stop only the assigned job on execution faults."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from monitor_exp2_6_cr import snapshot


def command(args: list[str]) -> str:
    """Read a bounded service/container status without an interactive shell."""
    result = subprocess.run(args, capture_output=True, text=True, timeout=45, check=False)
    return result.stdout.strip()


def main() -> None:
    """Poll once per minute; preserve evidence and stop rather than blindly retry faults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--unit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.unit.startswith("cr26-") or "/" in args.unit:
        parser.error("Require an assigned cr26- systemd unit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    previous, last_action, changed_at = None, None, time.monotonic()
    started = time.monotonic()
    while True:
        state = snapshot(args.root)
        state.update(checked_at=time.time(), unit=args.unit)
        state["service"] = command(["systemctl", "is-active", args.unit])
        faults = state.setdefault("faults", [])
        progress = (state.get("run"), state.get("actions"), state.get("model_installs"))
        if progress != last_action:
            last_action, changed_at = progress, time.monotonic()
        if state.get("phase") == "executing" and time.monotonic() - changed_at > 600:
            faults.append("no-action-or-model-progress-for-ten-minutes")
        free = shutil.disk_usage(args.output.parent).free
        state["disk_free_gib"] = round(free / 2**30, 1)
        if free < 10 * 2**30:
            faults.append("less-than-ten-GiB-free-disk")
        if state["service"] not in {"active", "activating"} and state.get("status") != "completed":
            faults.append("controller-service-not-active")
        if state.get("phase") in {"preparing", "building-or-starting"} and time.monotonic() - changed_at > 1200:
            faults.append("startup-exceeded-twenty-minutes")
        if time.monotonic() - started > 7 * 3600:
            faults.append("five-run-shard-exceeded-seven-hours")
        state["needs_inspection"] = bool(faults) or state.get("needs_inspection", False)
        if faults:
            # Freeze the controller first so it cannot advance to another run.
            command(["systemctl", "stop", args.unit])
            for identifier in state.get("containers", []):
                if identifier.startswith(("exp_cr26_agent_", "exp_cr26_env_")):
                    command(["docker", "stop", "--time", "30", identifier])
            state["heartbeat_action"] = "stopped-assigned-job"
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, allow_nan=False), encoding="utf-8")
        temporary.replace(args.output)
        event = (state.get("status"), state.get("run"), bool(faults))
        if event != previous:
            print(json.dumps(state, allow_nan=False), flush=True)
            previous = event
        if faults or state.get("status") == "completed":
            return
        time.sleep(60)


if __name__ == "__main__":
    main()
