"""
visualization.py

Visualization utilities for adversarial patch evaluation.

Author:
    Rishab Shetty
"""

from pathlib import Path
import cv2


def save_detection_image(result, output_path):
    """
    Save YOLO annotated image.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = result.plot()

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(output_path), image)

    print(f"✓ Saved {output_path}")