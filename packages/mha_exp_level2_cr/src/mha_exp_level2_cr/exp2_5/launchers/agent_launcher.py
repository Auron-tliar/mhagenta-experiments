"""Start the frozen-policy agent after its allowance measured from container launch."""

import asyncio
import os
import time
from typing import Any

import dill
from mhagenta.core.processes.mha_root import MHARoot


async def main() -> None:
    """Rebase the serialized start while preserving the complete execution budget."""

    with open("/agent/agent_params", "rb") as stream:
        params: dict[str, Any] = dill.load(stream)
    if os.environ.get("AGENT_ID"):
        params["agent_id"] = os.environ["AGENT_ID"]
    params["exec_start_time"] = time.time() + params.pop("container_start_delay")
    agent = MHARoot(**params)
    await agent.initialize()
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
