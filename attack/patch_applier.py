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
    # Internal helper
    # -------------------------------------------------------

    def _apply_single(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        x: int,
        y: int,
    ) -> torch.Tensor:
        """
        Apply a patch to a single image.

        image : (3,H,W)
        patch : (3,P,P)
        """

        patched = image.clone()

        _, H, W = patched.shape
        _, P, _ = patch.shape

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
    # Public API
    # -------------------------------------------------------

    def apply(
        self,
        image: torch.Tensor,
        patch: torch.Tensor,
        x: int,
        y: int,
    ) -> torch.Tensor:
        """
        Apply a patch to either

        Single Image:
            (3,H,W)

        or

        Batch:
            (B,3,H,W)
        """

        if image.ndim == 3:

            return self._apply_single(
                image,
                patch,
                x,
                y,
            )

        elif image.ndim == 4:

            outputs = []

            for img in image:

                outputs.append(
                    self._apply_single(
                        img,
                        patch,
                        x,
                        y,
                    )
                )

            return torch.stack(outputs)

        else:

            raise ValueError(
                f"Expected image tensor of shape (3,H,W) or (B,3,H,W), got {image.shape}"
            )

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

    print("Single Image")
    print("Output Shape :", output.shape)

    batch = torch.zeros(4, 3, 640, 640)

    output = applier.apply(
        image=batch,
        patch=patch,
        x=100,
        y=120,
    )

    print()
    print("Batch")
    print("Output Shape :", output.shape)