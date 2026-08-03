"""
live_detection.py

Live YOLOv8 Webcam Detection.

Responsibilities
----------------
- Open webcam
- Load YOLOv8 detector
- Run real-time inference
- Display annotated detections
- Display FPS and object count
- Exit cleanly

Author:
    Rishab Shetty
"""

import time
import cv2

from attack.config import load_config
from attack.detector import YOLODetector

from physical.webcam import Webcam


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Live YOLOv8 Detection")
    print("=" * 60)

    # -----------------------------------------------------
    # Load Configuration
    # -----------------------------------------------------

    cfg = load_config(
        "attack/configs/default.yaml"
    )

    print("✓ Configuration Loaded")

    # -----------------------------------------------------
    # Initialize Detector
    # -----------------------------------------------------

    detector = YOLODetector(cfg)

    print("✓ Detector Initialized")

    # -----------------------------------------------------
    # Initialize Webcam
    # -----------------------------------------------------

    webcam = Webcam()

    webcam.open()

    print("✓ Webcam Initialized")

    print()

    print("Controls")
    print("-" * 40)
    print("Press 'Q' to quit.")
    print("-" * 40)

    print("=" * 60)

    previous_time = time.time()

    try:

        # -------------------------------------------------
        # Live Detection Loop
        # -------------------------------------------------

        while True:

            ret, frame = webcam.read()

            if not ret:

                print("Failed to capture webcam frame.")

                break

            # ---------------------------------------------
            # FPS Calculation
            # ---------------------------------------------

            current_time = time.time()

            fps = 1.0 / max(
                current_time - previous_time,
                1e-6,
            )

            previous_time = current_time

            # ---------------------------------------------
            # YOLO Inference
            # ---------------------------------------------

            results = detector.predict(frame)

            annotated_frame = results[0].plot()

            num_objects = len(
                results[0].boxes
            )

            # ---------------------------------------------
            # Overlay Information
            # ---------------------------------------------

            cv2.putText(
                annotated_frame,
                f"FPS : {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Objects : {num_objects}",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            # ---------------------------------------------
            # Display
            # ---------------------------------------------

            cv2.imshow(
                "Robust Adversarial Patch - Live Detection",
                annotated_frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):

                break

    finally:

        webcam.release()

        print("=" * 60)
        print("Live Detection Closed")
        print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()