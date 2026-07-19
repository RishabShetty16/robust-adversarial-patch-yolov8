"""
train.py

Main entry point for adversarial patch training.

Author:
    Rishab Shetty
"""

import torch

from attack.config import load_config
from attack.dataset import COCODataset
from attack.detector import YOLODetector
from attack.patch import AdversarialPatch
from attack.patch_applier import PatchApplier
from attack.trainer import PatchTrainer


def main():

    print("=" * 60)
    print("Adversarial Patch Training")
    print("=" * 60)

    # ------------------------------------
    # Load configuration
    # ------------------------------------

    cfg = load_config(
        "attack/configs/default.yaml"
    )

    # ------------------------------------
    # Dataset
    # ------------------------------------

    dataset = COCODataset(cfg)

    # ------------------------------------
    # Detector
    # ------------------------------------

    detector = YOLODetector(cfg)

    # ------------------------------------
    # Patch
    # ------------------------------------

    patch = AdversarialPatch(
        size=cfg["patch"]["size"],
        initialization=cfg["patch"]["initialization"],
        clamp_min=cfg["patch"]["clamp_min"],
        clamp_max=cfg["patch"]["clamp_max"],
    )

    # ------------------------------------
    # Optimizer
    # ------------------------------------

    optimizer = torch.optim.Adam(
        patch.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )

    # ------------------------------------
    # Patch Applier
    # ------------------------------------

    patch_applier = PatchApplier()

    # ------------------------------------
    # Trainer
    # ------------------------------------

    trainer = PatchTrainer(
        cfg,
        detector,
        dataset,
        patch,
        optimizer,
        patch_applier,
    )

    trainer.train()


if __name__ == "__main__":
    main()