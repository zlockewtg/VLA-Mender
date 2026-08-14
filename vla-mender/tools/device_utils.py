from __future__ import annotations

import torch


def activate_cuda_device(device: str, *, service_name: str) -> torch.device:
    """Bind a service process to a CUDA device before vendor code uses bare `.cuda()`."""
    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError(f"{service_name} requires a CUDA device, got {device!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{service_name} requested CUDA, but CUDA is not available")

    index = 0 if requested.index is None else requested.index
    device_count = torch.cuda.device_count()
    if index < 0 or index >= device_count:
        raise ValueError(
            f"CUDA device index {index} is out of range for {device_count} visible device(s)"
        )

    torch.cuda.set_device(index)
    active = torch.device("cuda", torch.cuda.current_device())
    if active.index != index:
        raise RuntimeError(f"Failed to activate {requested}; PyTorch reports {active}")
    return active
