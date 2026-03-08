"""Utility functions for device management and reproducibility."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np
import torch


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU).
    
    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device_info() -> dict[str, str]:
    """Get information about the current device.
    
    Returns:
        Dictionary containing device information
    """
    device = get_device()
    info = {"device": str(device)}
    
    if device.type == "cuda":
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory"] = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    elif device.type == "mps":
        info["mps_available"] = "True"
    else:
        info["cpu_cores"] = str(torch.get_num_threads())
    
    return info
