"""
comparison.py

Create side-by-side comparison images.

Author:
    Rishab Shetty
"""

from pathlib import Path
import cv2
import numpy as np


def create_comparison(
    original_path,
    patched_path,
    output_path,
):
    """
    Create a side-by-side comparison image.
    """

    original = cv2.imread(original_path)
    patched = cv2.imread(patched_path)

    if original is None:
        raise FileNotFoundError(original_path)

    if patched is None:
        raise FileNotFoundError(patched_path)

    height = max(original.shape[0], patched.shape[0])

    if original.shape[0] != height:
        original = cv2.resize(
            original,
            (
                int(original.shape[1] * height / original.shape[0]),
                height,
            ),
        )

    if patched.shape[0] != height:
        patched = cv2.resize(
            patched,
            (
                int(patched.shape[1] * height / patched.shape[0]),
                height,
            ),
        )

    comparison = np.hstack([original, patched])

    cv2.putText(
        comparison,
        "Original",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    cv2.putText(
        comparison,
        "Patched",
        (original.shape[1] + 20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        comparison,
    )

    print(f"✓ Saved {output_path}")