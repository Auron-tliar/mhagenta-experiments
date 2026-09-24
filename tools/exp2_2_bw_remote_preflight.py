"""Fail-fast GPU/runtime checks for a paid 2-2-BW G7 execution."""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

import mhagenta
from mhagenta import Orchestrator


DEFAULT_IMAGE = "aurontliar/mhagenta:1.4.12-torch13.0"
REQUIRED_MHAGENTA_VERSION = "1.4.12"
REQUIRED_TORCH_VERSION = "2.14.0"
MINIMUM_G7_DRIVER_MAJOR = 595


def _run(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        suffix = detail[-1] if detail else "no diagnostic output"
        raise RuntimeError(f"{command[0]} failed: {suffix}")
    return result.stdout.strip()


def _host_gpus() -> list[dict[str, Any]]:
    output = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    rows = []
    for line in output.splitlines():
        index, name, memory_mib, driver = (part.strip() for part in line.split(",", 3))
        rows.append({
            "index": int(index),
            "name": name,
            "memory_mib": int(memory_mib),
            "driver_version": driver,
        })
    return rows


def _container_probe(image: str, gpu_device: int) -> dict[str, Any]:
    program = (
        "import json,torch; "
        "x=torch.ones((16,16),device='cuda:0'); result=(x@x).sum().item(); "
        "print(json.dumps({'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),"
        "'device_name':torch.cuda.get_device_name(0),'cuda_compute_result':result}))"
    )
    output = _run([
        "docker", "run", "--rm", "--gpus", f"device={gpu_device}",
        image, "python", "-c", program,
    ])
    return json.loads(output.splitlines()[-1])


def _orchestrator_probe(gpu_device: int) -> dict[str, Any]:
    """Verify that the imported framework can create the required GPU request."""

    if mhagenta.__version__ != REQUIRED_MHAGENTA_VERSION:
        raise RuntimeError(
            f"Expected MHAgentA {REQUIRED_MHAGENTA_VERSION}, found {mhagenta.__version__}"
        )
    requests = Orchestrator._resolve_gpu_ids([str(gpu_device)])
    if requests is None or len(requests) != 1:
        raise RuntimeError("The standard Orchestrator did not create one GPU request")
    request = requests[0]
    if request.get("DeviceIDs") != [str(gpu_device)]:
        raise RuntimeError("The standard Orchestrator did not preserve the assigned GPU ID")
    if request.get("Capabilities") != [["gpu"]]:
        raise RuntimeError("The standard Orchestrator GPU request has invalid capabilities")
    return {
        "mhagenta_version": mhagenta.__version__,
        "device_ids": request["DeviceIDs"],
        "capabilities": request["Capabilities"],
    }


def preflight(gpu_device: int, image: str) -> dict[str, Any]:
    """Return machine-readable evidence or raise before training can start."""

    gpus = _host_gpus()
    if len(gpus) != 2:
        raise RuntimeError(f"Expected the two GPUs of g7.12xlarge, found {len(gpus)}")
    selected = next((gpu for gpu in gpus if gpu["index"] == gpu_device), None)
    if selected is None:
        raise RuntimeError(f"GPU {gpu_device} is not present")
    if int(str(selected["driver_version"]).split(".", 1)[0]) < MINIMUM_G7_DRIVER_MAJOR:
        raise RuntimeError(
            f"G7 requires NVIDIA driver {MINIMUM_G7_DRIVER_MAJOR} or newer; "
            f"found {selected['driver_version']}"
        )
    if selected["memory_mib"] < 30_000:
        raise RuntimeError(f"GPU {gpu_device} exposes less than 30,000 MiB")

    orchestrator = _orchestrator_probe(gpu_device)
    container = _container_probe(image, gpu_device)
    if str(container.get("torch", "")).split("+", 1)[0] != REQUIRED_TORCH_VERSION:
        raise RuntimeError(f"The agent image requires Torch {REQUIRED_TORCH_VERSION}")
    if not container.get("cuda_available") or container.get("device_count") != 1:
        raise RuntimeError("The agent image does not see exactly its assigned CUDA GPU")
    if not container.get("device_name") or container.get("cuda_compute_result") != 4096.0:
        raise RuntimeError("The agent image did not confirm its GPU name and CUDA computation")
    return {
        "status": "passed",
        "assigned_gpu": gpu_device,
        "minimum_driver_major": MINIMUM_G7_DRIVER_MAJOR,
        "host_gpus": gpus,
        "image": image,
        "standard_orchestrator": orchestrator,
        "container": container,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    args = parser.parse_args()
    print(json.dumps(preflight(args.gpu_device, args.image), indent=2))


if __name__ == "__main__":
    main()
