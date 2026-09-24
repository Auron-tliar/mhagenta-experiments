"""Start the one-hour agent after a fresh 60-second container startup allowance."""

import asyncio
import os
import time

import dill
from mhagenta.core.processes.mha_root import MHARoot


async def main() -> None:
    """Rebase the serialized build-time start; preserve the full execution duration."""
    with open("/agent/agent_params", "rb") as stream:
        params = dill.load(stream)
    if os.environ.get("AGENT_ID"):
        params["agent_id"] = os.environ["AGENT_ID"]
    params["exec_start_time"] = time.time() + 60
    agent = MHARoot(**params)
    await agent.initialize()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
