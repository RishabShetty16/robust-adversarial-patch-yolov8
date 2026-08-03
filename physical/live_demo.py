"""
live_demo.py

Physical demonstration.

Displays:

- Live webcam feed
- Learned adversarial patch

Author:
    Rishab Shetty
"""

import cv2
import time

from physical.webcam import Webcam
from physical.display import PatchDisplay


def main():

    print("=" * 60)
    print("Physical Demo")
    print("=" * 60)

    # -------------------------------------------------
    # Initialize Webcam
    # -------------------------------------------------

    webcam = Webcam()

    webcam.open()

    # -------------------------------------------------
    # Load Patch
    # -------------------------------------------------

    display = PatchDisplay()

    display.load()

    patch_image = display.image()

    # Enlarge for visibility
    patch_image = cv2.resize(
        patch_image,
        (512, 512),
        interpolation=cv2.INTER_NEAREST,
    )

    print()

    print("Controls")
    print("-" * 40)
    print("Press 'Q' to quit the demo.")
    print("-" * 40)

    print()

    print("=" * 60)

    # -------------------------------------------------
    # Demo Loop
    # -------------------------------------------------
    
    previous_time = time.time()

    while True:

        ret, frame = webcam.read()
        current_time = time.time()

        fps = 1.0 / max(
            current_time - previous_time,
            1e-6,
        )

        previous_time = current_time

        if not ret:
            print("Failed to read webcam frame.")
            break
        
        cv2.putText(
            frame,
            f"FPS : {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow(
            "Robust Adversarial Patch - Webcam",
            frame,
        )

        cv2.imshow(
            "Robust Adversarial Patch",
            patch_image,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    webcam.release()


if __name__ == "__main__":
    main()