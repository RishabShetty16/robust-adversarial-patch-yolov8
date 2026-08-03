"""
display.py

Patch display utilities.

Responsibilities
----------------
- Load trained adversarial patch
- Convert patch to image
- Display patch
- Save patch image if required

Author:
    Rishab Shetty
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from attack.patch import AdversarialPatch


class PatchDisplay:

    def __init__(
        self,
        checkpoint_path="outputs/checkpoints/best.pt",
    ):

        self.checkpoint_path = Path(checkpoint_path)

        self.patch = None

    # -------------------------------------------------
    # Load Patch
    # -------------------------------------------------

    def load(self):

        print("=" * 60)
        print("Loading Patch")
        print("=" * 60)

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location="cpu",
        )

        self.patch = AdversarialPatch()

        self.patch.load_state_dict(
            checkpoint["patch"]
        )

        print("Checkpoint :", self.checkpoint_path)
        print("Epoch      :", checkpoint["epoch"])
        print("Best Loss  :", checkpoint["best_loss"])

        print("=" * 60)

    # -------------------------------------------------
    # Convert to OpenCV Image
    # -------------------------------------------------

    def image(self):

        image = (
            self.patch()
            .detach()
            .cpu()
            .permute(1, 2, 0)
            .numpy()
        )

        image = np.clip(
            image,
            0,
            1,
        )

        image = (255 * image).astype(
            np.uint8
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

        return image

    # -------------------------------------------------
    # Show Patch
    # -------------------------------------------------

    def show(self):

        image = self.image()

        # Enlarge for visualization
        image = cv2.resize(
            image,
            (512, 512),
            interpolation=cv2.INTER_NEAREST,
        )

        while True:

            cv2.imshow(
                "Adversarial Patch",
                image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

        cv2.destroyAllWindows()


# =====================================================
# Module Test
# =====================================================

if __name__ == "__main__":

    display = PatchDisplay()

    display.load()

    display.show()