"""
eot.py

Expectation Over Transformation (EOT)

Applies random transformations to the adversarial patch.
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF


class EOT:

    def __init__(self, cfg):

        self.cfg = cfg

        self.enabled = cfg["eot"]["enabled"]
        self.max_rotation = cfg["eot"]["rotation"]["degrees"]

    def __call__(self, patch: torch.Tensor):

        if not self.enabled:
            return patch

        angle = random.uniform(
            -self.max_rotation,
            self.max_rotation,
        )

        patch = TF.rotate(
            patch,
            angle,
            interpolation=TF.InterpolationMode.BILINEAR,
        )

        patch = patch.clamp(0.0, 1.0)

        return patch

    def __repr__(self):

        return "EOT(rotation)"