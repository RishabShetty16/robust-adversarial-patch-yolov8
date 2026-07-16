"""
config.py

Configuration management for the Robust Adversarial Patch project.

Responsibilities
----------------
1. Load experiment configuration from a YAML file.
2. Validate required configuration sections.
3. Select the appropriate computation device.

Author:
    Rishab Shetty

Project:
    Robust Physical Adversarial Patch Attack for YOLOv8
"""

from pathlib import Path
from typing import Any, Dict

import torch
import yaml


# ---------------------------------------------------------------------
# Required configuration sections
# ---------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "experiment",
    "seed",
    "device",
    "dataset",
    "model",
    "patch",
    "optimizer",
    "attack",
    "eot",
    "logging",
]


# ---------------------------------------------------------------------
# Load configuration
# ---------------------------------------------------------------------

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML configuration file.

    Parameters
    ----------
    config_path : str | Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    ValueError
        If the YAML file is empty.
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found:\n{config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty:\n{config_path}"
        )

    validate_config(config)

    return config


# ---------------------------------------------------------------------
# Validate configuration
# ---------------------------------------------------------------------

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required sections in the configuration.

    Parameters
    ----------
    config : dict
        Loaded YAML configuration.

    Raises
    ------
    KeyError
        If a required section is missing.
    """

    for section in REQUIRED_SECTIONS:

        if section not in config:

            raise KeyError(
                f"Missing required configuration section: '{section}'"
            )


# ---------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------

def get_device(config: Dict[str, Any]) -> torch.device:
    """
    Determine which device should be used.

    Supported options
    -----------------
    auto
    cpu
    cuda

    Returns
    -------
    torch.device
    """

    device = config["device"].lower()

    if device == "auto":

        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if device == "cuda":

        if not torch.cuda.is_available():

            print(
                "[WARNING] CUDA requested but unavailable."
                " Falling back to CPU."
            )

            return torch.device("cpu")

        return torch.device("cuda")

    if device == "cpu":

        return torch.device("cpu")

    raise ValueError(
        f"Unsupported device option: {device}"
    )


# ---------------------------------------------------------------------
# Pretty configuration summary
# ---------------------------------------------------------------------

def print_config_summary(config: Dict[str, Any]) -> None:
    """
    Print a concise experiment summary.
    """

    print("=" * 60)
    print("Experiment Configuration")
    print("=" * 60)

    print(f"Experiment : {config['experiment']['name']}")
    print(f"Device     : {config['device']}")
    print(f"Model      : {config['model']['weights']}")
    print(f"Patch Size : {config['patch']['size']}")
    print(f"Optimizer  : {config['optimizer']['name']}")
    print(f"LR         : {config['optimizer']['lr']}")
    print(f"Iterations : {config['optimizer']['iterations']}")
    print(f"Attack     : {config['attack']['mode']}")

    print("=" * 60)