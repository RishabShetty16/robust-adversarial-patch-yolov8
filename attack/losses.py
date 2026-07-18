"""
losses.py

Loss functions for adversarial patch optimization.

Responsibilities
----------------
- Baseline optimization loss
- Person suppression loss
- Confidence loss
- Objectness loss
- Total Variation (TV) loss
- Non-Printability Score (NPS) placeholder

Author:
    Rishab Shetty
"""

from __future__ import annotations

import torch


# ==========================================================
# Baseline Optimization Loss
# ==========================================================

def baseline_loss(predictions: torch.Tensor) -> torch.Tensor:
    """
    Temporary differentiable loss used to verify the
    optimization pipeline.

    NOTE
    ----
    This is NOT the final adversarial objective.
    """

    return predictions.mean()


# ==========================================================
# Person Suppression Loss
# ==========================================================

def person_suppression_loss(
    person_scores: torch.Tensor,
) -> torch.Tensor:
    """
    Suppress detections of the target class.

    Parameters
    ----------
    person_scores : Tensor
        Shape (B, N)

    Returns
    -------
    torch.Tensor
        Mean confidence of all person predictions.

    Lower is better.
    """

    if person_scores.numel() == 0:
        return torch.tensor(
            0.0,
            device=person_scores.device,
            requires_grad=True,
        )

    return person_scores.mean()


# ==========================================================
# Confidence Loss
# ==========================================================

def confidence_loss(confidences: torch.Tensor) -> torch.Tensor:
    """
    Reduce detector confidence.
    """

    if confidences.numel() == 0:
        return torch.tensor(
            0.0,
            device=confidences.device,
            requires_grad=True,
        )

    return confidences.mean()


# ==========================================================
# Objectness Loss
# ==========================================================

def objectness_loss(objectness: torch.Tensor) -> torch.Tensor:
    """
    Reduce objectness score.

    Placeholder for detectors that expose
    an explicit objectness output.
    """

    if objectness.numel() == 0:
        return torch.tensor(
            0.0,
            device=objectness.device,
            requires_grad=True,
        )

    return objectness.mean()


# ==========================================================
# Total Variation Loss
# ==========================================================

def total_variation_loss(
    patch: torch.Tensor,
) -> torch.Tensor:
    """
    Encourage smooth adversarial patches.
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

def non_printability_score(
    patch: torch.Tensor,
) -> torch.Tensor:
    """
    Placeholder for the Non-Printability
    Score (NPS).
    """

    return torch.tensor(
        0.0,
        device=patch.device,
    )


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    predictions = torch.randn(
        1,
        84,
        8400,
        requires_grad=True,
    )

    person_scores = torch.rand(
        1,
        8400,
        requires_grad=True,
    )

    confidences = torch.tensor(
        [0.91, 0.84, 0.63],
        requires_grad=True,
    )

    objectness = torch.tensor(
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
        "Baseline Loss :",
        baseline_loss(predictions).item(),
    )

    print(
        "Person Suppression Loss :",
        person_suppression_loss(person_scores).item(),
    )

    print(
        "Confidence Loss :",
        confidence_loss(confidences).item(),
    )

    print(
        "Objectness Loss :",
        objectness_loss(objectness).item(),
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