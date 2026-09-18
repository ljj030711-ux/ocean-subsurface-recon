"""Centralized reproducibility controls and runtime traceability metadata."""

import os
import platform
import random
import sys

import numpy as np
import torch


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def configure_reproducibility(seed):
    """Seed all project RNGs and require deterministic PyTorch algorithms.

    This function should run before model construction and DataLoader creation.
    PyTorch will raise instead of silently using a known non-deterministic
    operation, which makes a failed reproducibility guarantee visible.
    """
    seed = int(seed)
    if seed < 0:
        raise ValueError("随机种子必须是非负整数")

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # CUDA bicubic interpolation backward has no deterministic implementation
    # in the training environment. Keep deterministic algorithms enabled for
    # supported operations, but warn instead of aborting for that kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # TF32 is deterministic on a fixed GPU, but disabling it makes the
    # numerical mode explicit and prevents an unnoticed precision-mode change.
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn") and hasattr(
        torch.backends.cudnn, "allow_tf32"
    ):
        torch.backends.cudnn.allow_tf32 = False


def seed_dataloader_worker(_worker_id):
    """Seed Python and NumPy inside a PyTorch DataLoader worker process."""
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader_generator(seed):
    """Return an independent generator for deterministic sampling/worker seeds."""
    return torch.Generator().manual_seed(int(seed))


def build_reproducibility_report(seed, device):
    """Build JSON-safe seed, software, and accelerator provenance metadata."""
    device = torch.device(device)
    device_index = device.index
    device_name = platform.processor() or platform.machine() or "unknown"
    device_count = 1
    total_memory_bytes = None
    compute_capability = None

    if device.type == "cuda":
        device_index = (
            torch.cuda.current_device() if device_index is None else device_index
        )
        properties = torch.cuda.get_device_properties(device_index)
        device_name = properties.name
        device_count = torch.cuda.device_count()
        total_memory_bytes = int(properties.total_memory)
        major, minor = torch.cuda.get_device_capability(device_index)
        compute_capability = f"{major}.{minor}"
    elif device.type == "mps":
        device_name = f"Apple Metal ({platform.machine() or 'unknown'})"

    cudnn_version = None
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        cudnn_version = torch.backends.cudnn.version()

    cuda_matmul_tf32 = None
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        cuda_matmul_tf32 = bool(torch.backends.cuda.matmul.allow_tf32)
    cudnn_tf32 = None
    if hasattr(torch.backends, "cudnn") and hasattr(
        torch.backends.cudnn, "allow_tf32"
    ):
        cudnn_tf32 = bool(torch.backends.cudnn.allow_tf32)

    return {
        "seed": int(seed),
        "strict_determinism": False,
        "determinism_policy": "warn_only",
        "known_nondeterministic_operation": (
            "upsample_bicubic2d_backward_out_cuda"
        ),
        "deterministic_algorithms": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "rngs": ["python", "numpy", "torch_cpu", "torch_cuda_all"],
        "dataloader_worker_seeding": "torch_initial_seed_mod_2^32",
        "environment": {
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get(
                "CUBLAS_WORKSPACE_CONFIG"
            ),
        },
        "backend": {
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cuda_matmul_allow_tf32": cuda_matmul_tf32,
            "cudnn_allow_tf32": cudnn_tf32,
        },
        "device": {
            "type": device.type,
            "torch_device": str(device),
            "index": device_index,
            "name": device_name,
            "visible_device_count": int(device_count),
            "total_memory_bytes": total_memory_bytes,
            "compute_capability": compute_capability,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            # PyTorch 2.6 may expose this as a TorchVersion (a str subclass).
            # Store a plain string so weights_only checkpoint loading needs no
            # custom allowlist for checkpoints created after this fix.
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "cuda": torch.version.cuda,
            "cudnn": cudnn_version,
        },
    }


def print_reproducibility_report(report):
    """Print the most important provenance fields for terminal log capture."""
    device = report["device"]
    software = report["software"]
    print(f"随机种子:   {report['seed']}")
    print(
        "确定性模式: warn-only "
        "(已固定随机种子并关闭cuDNN benchmark；"
        "bicubic CUDA backward可能产生非逐位一致结果)"
    )
    print(
        f"运行设备:   {device['torch_device']} | {device['name']} | "
        f"可见设备数={device['visible_device_count']}"
    )
    print(
        f"软件环境:   Python {software['python']} | PyTorch {software['torch']} | "
        f"CUDA {software['cuda']} | cuDNN {software['cudnn']}"
    )
