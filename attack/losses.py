"""
losses.py

Loss functions for adversarial patch optimization.

Responsibilities
----------------
- Confidence loss
- Objectness loss
- Total Variation loss
- NPS placeholder

Author:
    Rishab Shetty
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ==========================================================
# Confidence Loss
# ==========================================================

def confidence_loss(confidences: torch.Tensor) -> torch.Tensor:
    """
    Reduce detector confidence.

    Lower confidence = better attack.
    """

    if confidences.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    return confidences.mean()


# ==========================================================
# Objectness Loss
# ==========================================================

def objectness_loss(objectness: torch.Tensor) -> torch.Tensor:
    """
    Reduce objectness score.
    """

    if objectness.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)

    return objectness.mean()


# ==========================================================
# Total Variation Loss
# ==========================================================

def total_variation_loss(patch: torch.Tensor) -> torch.Tensor:
    """
    Encourage smooth patches.
    """

    tv_h = torch.abs(
        patch[:, :, 1:] - patch[:, :, :-1]
    ).mean()

    tv_w = torch.abs(
        patch[:, 1:, :] - patch[:, :-1, :]
    ).mean()

    return tv_h + tv_w


# ==========================================================
# Non-Printability Score
# ==========================================================

def non_printability_score(patch: torch.Tensor) -> torch.Tensor:
    """
    Placeholder.

    Will be implemented during the
    physical attack stage.
    """

    return torch.tensor(
        0.0,
        device=patch.device,
    )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    conf = torch.tensor(
        [0.91, 0.84, 0.63],
        requires_grad=True,
    )

    obj = torch.tensor(
        [0.93, 0.82],
        requires_grad=True,
    )

    patch = torch.rand(
        3,
        160,
        160,
        requires_grad=True,
    )

    print("=" * 60)

    print("Testing Loss Functions")

    print("=" * 60)

    print(
        "Confidence Loss :",
        confidence_loss(conf).item(),
    )

    print(
        "Objectness Loss :",
        objectness_loss(obj).item(),
    )

    print(
        "TV Loss :",
        total_variation_loss(patch).item(),
    )

    print(
        "NPS :",
        non_printability_score(patch).item(),
    )

    print("=" * 60)