"""
patch.py

Adversarial Patch module.

This module defines the learnable adversarial patch used during
optimization. The patch is implemented as a PyTorch nn.Module so it
can be optimized using standard PyTorch optimizers.

Responsibilities
----------------
- Initialize the patch using multiple strategies
- Store the learnable tensor
- Clamp pixel values
- Save / load
- Visualize
- Compute statistics

Author:
    Rishab Shetty
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class AdversarialPatch(nn.Module):
    """
    Learnable adversarial patch.
    """

    SUPPORTED_INITIALIZATIONS = (
        "gray",
        "random",
        "checkerboard",
        "gaussian",
    )

    def __init__(
        self,
        size: int = 160,
        initialization: str = "gray",
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        super().__init__()

        self.size = size
        self.initialization = initialization

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.patch = nn.Parameter(
            self._initialize_patch()
        )

    # ---------------------------------------------------------
    # Individual Initialization Methods
    # ---------------------------------------------------------

    def _initialize_gray(self) -> torch.Tensor:

        return torch.full(
            (3, self.size, self.size),
            0.5,
            dtype=torch.float32,
        )

    def _initialize_random(self) -> torch.Tensor:

        return torch.rand(
            (3, self.size, self.size),
            dtype=torch.float32,
        )

    def _initialize_checkerboard(self) -> torch.Tensor:

        patch = torch.zeros(
            (3, self.size, self.size),
            dtype=torch.float32,
        )

        step = 16

        for i in range(self.size):
            for j in range(self.size):

                if ((i // step) + (j // step)) % 2 == 0:
                    patch[:, i, j] = 1.0

        return patch

    def _initialize_gaussian(self) -> torch.Tensor:

        patch = torch.randn(
            (3, self.size, self.size),
            dtype=torch.float32,
        )

        patch = (patch - patch.min()) / (
            patch.max() - patch.min()
        )

        return patch

    # ---------------------------------------------------------
    # Patch Initialization
    # ---------------------------------------------------------

    def _initialize_patch(self) -> torch.Tensor:

        if self.initialization == "gray":
            return self._initialize_gray()

        elif self.initialization == "random":
            return self._initialize_random()

        elif self.initialization == "checkerboard":
            return self._initialize_checkerboard()

        elif self.initialization == "gaussian":
            return self._initialize_gaussian()

        raise ValueError(
            f"Unknown initialization '{self.initialization}'. "
            f"Supported values: {self.SUPPORTED_INITIALIZATIONS}"
        )

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self) -> torch.Tensor:

        return self.patch

    # ---------------------------------------------------------
    # Clamp
    # ---------------------------------------------------------

    @torch.no_grad()
    def clamp(self):

        self.patch.clamp_(
            self.clamp_min,
            self.clamp_max,
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    @torch.no_grad()
    def statistics(self):

        return {

            "shape": tuple(self.patch.shape),

            "mean": self.patch.mean().item(),

            "std": self.patch.std().item(),

            "min": self.patch.min().item(),

            "max": self.patch.max().item(),

        }

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    def save(self, path):

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            self.state_dict(),
            path,
        )

    # ---------------------------------------------------------
    # Load
    # ---------------------------------------------------------

    def load(self, path):

        state = torch.load(
            path,
            map_location="cpu",
        )

        self.load_state_dict(state)

    # ---------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------

    @torch.no_grad()
    def visualize(
        self,
        save_path=None,
        show=False,
    ):

        image = (
            self.patch
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        plt.figure(figsize=(5, 5))

        plt.imshow(image)

        plt.axis("off")

        plt.tight_layout()

        if save_path is not None:

            save_path = Path(save_path)

            save_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            plt.savefig(
                save_path,
                dpi=300,
                bbox_inches="tight",
            )

        if show:
            plt.show()

        plt.close()

    # ---------------------------------------------------------
    # Pretty Printing
    # ---------------------------------------------------------

    def extra_repr(self):

        return (
            f"size={self.size}, "
            f"initialization='{self.initialization}'"
        )


# =============================================================
# Module Test
# =============================================================

if __name__ == "__main__":

    for mode in AdversarialPatch.SUPPORTED_INITIALIZATIONS:

        print("=" * 60)
        print(f"Testing Initialization : {mode}")
        print("=" * 60)

        patch = AdversarialPatch(
            size=160,
            initialization=mode,
        )

        print(patch)

        print("\nStatistics")

        stats = patch.statistics()

        for key, value in stats.items():

            print(f"{key}: {value}")

        patch.clamp()

        patch.save(
            f"outputs/patches/{mode}_patch.pt"
        )

        patch.visualize(
            save_path=f"outputs/patches/{mode}_patch.png"
        )

        print(f"\n✓ {mode.capitalize()} patch saved successfully.\n")