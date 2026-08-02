"""
eot.py

Expectation Over Transformation (EOT)

Applies random transformations to the adversarial patch.

Current Transformations
-----------------------
- Random Rotation
- Random Scaling
- Random Brightness
- Random Contrast
- Gaussian Noise
- Gaussian Blur

Author:
    Rishab Shetty
"""

from __future__ import annotations

import random

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class EOT:
    """
    Expectation Over Transformation (EOT).

    Randomly transforms the adversarial patch during
    training to improve robustness.
    """

    def __init__(self, cfg):

        self.cfg = cfg

        self.enabled = cfg["eot"]["enabled"]

        # ---------------------------------------------------
        # Rotation
        # ---------------------------------------------------

        self.rotation_enabled = (
            cfg["eot"]["rotation"]["enabled"]
        )

        self.max_rotation = (
            cfg["eot"]["rotation"]["degrees"]
        )

        # ---------------------------------------------------
        # Scaling
        # ---------------------------------------------------

        self.scale_enabled = (
            cfg["eot"]["scale"]["enabled"]
        )

        self.min_scale = (
            cfg["eot"]["scale"]["min"]
        )

        self.max_scale = (
            cfg["eot"]["scale"]["max"]
        )

        # ---------------------------------------------------
        # Brightness
        # ---------------------------------------------------

        brightness_cfg = cfg["eot"].get(
            "brightness",
            {},
        )

        self.brightness_enabled = (
            brightness_cfg.get(
                "enabled",
                False,
            )
        )

        self.min_brightness = (
            brightness_cfg.get(
                "min",
                1.0,
            )
        )

        self.max_brightness = (
            brightness_cfg.get(
                "max",
                1.0,
            )
        )

        # ---------------------------------------------------
        # Contrast
        # ---------------------------------------------------

        contrast_cfg = cfg["eot"].get(
            "contrast",
            {},
        )

        self.contrast_enabled = (
            contrast_cfg.get(
                "enabled",
                False,
            )
        )

        self.min_contrast = (
            contrast_cfg.get(
                "min",
                1.0,
            )
        )

        self.max_contrast = (
            contrast_cfg.get(
                "max",
                1.0,
            )
        )

        # ---------------------------------------------------
        # Gaussian Noise
        # ---------------------------------------------------

        noise_cfg = cfg["eot"].get(
            "noise",
            {},
        )

        self.noise_enabled = (
            noise_cfg.get(
                "enabled",
                False,
            )
        )

        self.noise_std = (
            noise_cfg.get(
                "std",
                0.03,
            )
        )

        # ---------------------------------------------------
        # Gaussian Blur
        # ---------------------------------------------------

        blur_cfg = cfg["eot"].get(
            "blur",
            {},
        )

        self.blur_enabled = (
            blur_cfg.get(
                "enabled",
                False,
            )
        )

        self.blur_kernel = (
            blur_cfg.get(
                "kernel_size",
                3,
            )
        )
    # -------------------------------------------------------
    # Random Rotation
    # -------------------------------------------------------

    def random_rotate(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:

        angle = random.uniform(
            -self.max_rotation,
            self.max_rotation,
        )

        patch = TF.rotate(
            patch,
            angle,
            interpolation=InterpolationMode.BILINEAR,
        )

        return patch

    # -------------------------------------------------------
    # Random Scaling
    # -------------------------------------------------------

    def random_scale(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:

        original_size = patch.shape[-1]

        scale = random.uniform(
            self.min_scale,
            self.max_scale,
        )

        new_size = max(
            1,
            int(original_size * scale),
        )

        patch = TF.resize(
            patch,
            [new_size, new_size],
            interpolation=InterpolationMode.BILINEAR,
        )

        if new_size < original_size:

            pad_total = original_size - new_size

            left = pad_total // 2
            right = pad_total - left

            top = pad_total // 2
            bottom = pad_total - top

            patch = TF.pad(
                patch,
                [left, top, right, bottom],
            )

        elif new_size > original_size:

            patch = TF.center_crop(
                patch,
                [original_size, original_size],
            )

        return patch

    # -------------------------------------------------------
    # Random Brightness
    # -------------------------------------------------------

    def random_brightness(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:

        factor = random.uniform(
            self.min_brightness,
            self.max_brightness,
        )

        return TF.adjust_brightness(
            patch,
            factor,
        )

    # -------------------------------------------------------
    # Random Contrast
    # -------------------------------------------------------

    def random_contrast(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:

        factor = random.uniform(
            self.min_contrast,
            self.max_contrast,
        )

        return TF.adjust_contrast(
            patch,
            factor,
        )

    # -------------------------------------------------------
    # Gaussian Noise
    # -------------------------------------------------------

    def random_noise(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Gaussian noise.
        """

        noise = (
            torch.randn_like(patch)
            * self.noise_std
        )

        patch = patch + noise

        return patch

    # -------------------------------------------------------
    # Gaussian Blur
    # -------------------------------------------------------

    def random_blur(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply Gaussian blur.
        """

        patch = TF.gaussian_blur(
            patch,
            kernel_size=self.blur_kernel,
        )

        return patch
    # -------------------------------------------------------
    # Apply EOT
    # -------------------------------------------------------

    def __call__(
        self,
        patch: torch.Tensor,
    ) -> torch.Tensor:

        if not self.enabled:
            return patch

        # Rotation
        if self.rotation_enabled:
            patch = self.random_rotate(patch)

        # Scaling
        if self.scale_enabled:
            patch = self.random_scale(patch)

        # Brightness
        if self.brightness_enabled:
            patch = self.random_brightness(patch)

        # Contrast
        if self.contrast_enabled:
            patch = self.random_contrast(patch)

        # Gaussian Noise
        if self.noise_enabled:
            patch = self.random_noise(patch)

        # Gaussian Blur
        if self.blur_enabled:
            patch = self.random_blur(patch)

        # Keep pixel values within valid range
        patch = patch.clamp(0.0, 1.0)

        return patch

    # -------------------------------------------------------
    # String Representation
    # -------------------------------------------------------

    def __repr__(self):

        if not self.enabled:
            return "EOT(disabled)"

        transforms = []

        if self.rotation_enabled:
            transforms.append("rotation")

        if self.scale_enabled:
            transforms.append("scaling")

        if self.brightness_enabled:
            transforms.append("brightness")

        if self.contrast_enabled:
            transforms.append("contrast")

        if self.noise_enabled:
            transforms.append("noise")

        if self.blur_enabled:
            transforms.append("blur")

        if len(transforms) == 0:
            return "EOT(no transforms enabled)"

        return f"EOT({', '.join(transforms)})"