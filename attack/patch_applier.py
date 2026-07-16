"""
patch_applier.py

Applies an adversarial patch onto images.

Responsibilities
----------------
- Apply a patch to one image
- Apply a patch to a batch of images
- Keep image values in valid range

Author:
    Rishab Shetty
"""

from __future__ import annotations

import torch


class PatchApplier:
    """
    Applies adversarial patches to images.
    """

    def __init__(self):

        pass

    # -------------------------------------------------------
    # Apply to one image
    # -------------------------------------------------------

    def apply(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        x: int,
        y: int,
    ) -> torch.Tensor:
        """
        Apply a patch to a single image.

        Parameters
        ----------
        image : Tensor
            Shape (3,H,W)

        patch : Tensor
            Shape (3,P,P)

        x : int
            Left coordinate

        y : int
            Top coordinate
        """

        patched = image.clone()

        _, H, W = patched.shape

        _, P, _ = patch.shape

        # Keep patch inside image

        x = max(0, min(x, W - P))
        y = max(0, min(y, H - P))

        patched[
            :,
            y:y + P,
            x:x + P,
        ] = patch

        patched.clamp_(0.0, 1.0)

        return patched

    # -------------------------------------------------------
    # Apply to batch
    # -------------------------------------------------------

    def apply_batch(
        self,
        images: torch.Tensor,
        patch: torch.Tensor,
        x: int,
        y: int,
    ) -> torch.Tensor:
        """
        Apply patch to a batch.

        Parameters
        ----------
        images : Tensor

            Shape (B,3,H,W)
        """

        outputs = []

        for image in images:

            outputs.append(
                self.apply(
                    image,
                    patch,
                    x,
                    y,
                )
            )

        return torch.stack(outputs)

    # -------------------------------------------------------

    def __repr__(self):

        return "PatchApplier()"


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    image = torch.zeros(3, 640, 640)

    patch = torch.ones(3, 160, 160) * 0.5

    applier = PatchApplier()

    output = applier.apply(
        image=image,
        patch=patch,
        x=100,
        y=120,
    )

    print(applier)

    print()

    print("Output Shape :", output.shape)

    print("Min :", output.min().item())

    print("Max :", output.max().item())