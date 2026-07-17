"""
Stage A Training

This experiment performs adversarial patch optimization.

Author:
    Rishab Shetty
"""

from pathlib import Path

import torch
import torch.optim as optim
from attack.config import load_config

from attack.dataset import COCODataset

from attack.patch import AdversarialPatch

from attack.detector import YOLODetector

from attack.patch_applier import PatchApplier

from attack.trainer import PatchTrainer

import torch


# ==========================================================
# Configuration
# ==========================================================

cfg = load_config("attack/configs/default.yaml")


# ==========================================================
# Dataset
# ==========================================================

dataset = COCODataset(cfg)


# ==========================================================
# Patch
# ==========================================================

patch = AdversarialPatch(
    size=cfg["patch"]["size"]
)


# ==========================================================
# Optimizer
# ==========================================================

optimizer = torch.optim.Adam(
    [patch.patch],
    lr=cfg["optimizer"]["lr"],
)


# ==========================================================
# Detector
# ==========================================================

detector = YOLODetector(cfg)


# ==========================================================
# Patch Applier
# ==========================================================

patch_applier = PatchApplier()

# ==========================================================
# Trainer
# ==========================================================

trainer = PatchTrainer(
    cfg=cfg,
    detector=detector,
    dataset=dataset,
    patch=patch,
    optimizer=optimizer,
    patch_applier=patch_applier,
)

trainer.train()