"""
utils.py

General utility functions for the Robust Adversarial Patch project.

Responsibilities
----------------
- Reproducibility
- Directory management
- Time utilities
- Model inspection

Author:
    Rishab Shetty
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


# ==========================================================
# Random Seed
# ==========================================================

def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Parameters
    ----------
    seed : int
        Random seed value.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================================
# Directory Helpers
# ==========================================================

def create_directory(path: str | Path) -> Path:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    path : str | Path

    Returns
    -------
    Path
    """

    path = Path(path)

    path.mkdir(parents=True, exist_ok=True)

    return path


def create_output_directories(base_dir: str | Path = "outputs") -> None:
    """
    Create all required output folders.

    Structure
    ---------

    outputs/
        checkpoints/
        patches/
        figures/
        logs/
    """

    base_dir = create_directory(base_dir)

    create_directory(base_dir / "checkpoints")
    create_directory(base_dir / "patches")
    create_directory(base_dir / "figures")
    create_directory(base_dir / "logs")


# ==========================================================
# Time
# ==========================================================

def get_timestamp() -> str:
    """
    Returns
    -------
    str

    Example
    -------
    20260716_214530
    """

    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ==========================================================
# Model Helpers
# ==========================================================

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """
    Count model parameters.

    Parameters
    ----------
    model : torch.nn.Module

    Returns
    -------
    total_parameters
    trainable_parameters
    """

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ==========================================================
# Module Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Testing utils.py")
    print("=" * 60)

    set_seed(42)

    create_output_directories()

    print("Timestamp :", get_timestamp())

    model = torch.nn.Linear(10, 5)

    total, trainable = count_parameters(model)

    print(f"Total Parameters     : {total}")
    print(f"Trainable Parameters : {trainable}")

    print("=" * 60)