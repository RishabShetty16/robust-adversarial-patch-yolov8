"""
saliency_defense.py

Saliency-based adversarial patch defense.

Responsibilities
----------------
- Generate saliency maps
- Highlight suspicious regions
- Compute defense score

Author:
    Rishab Shetty
"""

from __future__ import annotations

import cv2
import numpy as np


class SaliencyDefense:
    """
    Saliency-based defense module.
    """

    def __init__(self):

        self.success = False

        self.detector = None

        if hasattr(cv2, "saliency"):

            try:

                self.detector = (
                    cv2.saliency
                    .StaticSaliencyFineGrained_create()
                )

                self.success = True

            except Exception:

                self.detector = None

                self.success = False

    # -------------------------------------------------
    # Compute Saliency
    # -------------------------------------------------

    def compute_saliency(
        self,
        image,
    ):
        """
        Compute gradient-based saliency map.
        """

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

        grad_x = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        grad_y = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        magnitude = cv2.magnitude(
            grad_x,
            grad_y,
        )

        magnitude = cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )

        return magnitude.astype("uint8")

    # -------------------------------------------------
    # Generate Heatmap
    # -------------------------------------------------

    def heatmap(
        self,
        image,
    ):
        """
        Generate colored saliency heatmap.
        """

        saliency = self.compute_saliency(
            image,
        )

        heatmap = cv2.applyColorMap(
            saliency,
            cv2.COLORMAP_JET,
        )

        return heatmap

    # -------------------------------------------------
    # Overlay Heatmap
    # -------------------------------------------------

    def overlay(
        self,
        image,
        alpha=0.4,
    ):
        """
        Overlay saliency heatmap
        onto the original image.
        """

        heatmap = self.heatmap(
            image,
        )

        overlay = cv2.addWeighted(
            image,
            1.0 - alpha,
            heatmap,
            alpha,
            0,
        )

        return overlay

    