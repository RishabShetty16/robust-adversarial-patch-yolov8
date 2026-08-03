"""
attack_demo.py

Physical Adversarial Patch Demonstration.

Responsibilities
----------------
- Open webcam
- Load trained adversarial patch
- Load YOLOv8 detector
- Display live detections
- Display adversarial patch
- Demonstrate physical-world attack setup

Author:
    Rishab Shetty
"""

import time
import cv2

from attack.config import load_config
from attack.detector import YOLODetector

from physical.display import PatchDisplay
from physical.webcam import Webcam

from physical.metrics import PhysicalMetrics

# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Physical Attack Demonstration")
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

    print("✓ YOLOv8 Loaded")

    # -----------------------------------------------------
    # Initialize Webcam
    # -----------------------------------------------------

    webcam = Webcam()

    webcam.open()

    print("✓ Webcam Initialized")

    # -----------------------------------------------------
    # Load Trained Patch
    # -----------------------------------------------------

    display = PatchDisplay()

    display.load()
    metrics = PhysicalMetrics()

    print("✓ Trained Patch Loaded")

    # -----------------------------------------------------
    # Prepare Patch Image
    # -----------------------------------------------------

    patch_image = display.image()

    patch_image = cv2.resize(
        patch_image,
        (600, 600),
        interpolation=cv2.INTER_NEAREST,
    )

    print()

    print("Controls")
    print("-" * 40)
    print("Press 'Q' to quit.")
    print("-" * 40)

    print("=" * 60)

    previous_time = time.time()

    try:

        # -------------------------------------------------
        # Live Demo Loop
        # -------------------------------------------------

        while True:

            ret, frame = webcam.read()

            if not ret:

                print("Failed to capture webcam frame.")

                break

            # ---------------------------------------------
            # FPS
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
            metrics.update(
                results,
                fps,
            )

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
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                annotated_frame,
                f"Objects : {num_objects}",
                (10, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            # ---------------------------------------------
            # Display Windows
            # ---------------------------------------------

            cv2.imshow(
                "Physical Attack Demo",
                annotated_frame,
            )

            cv2.imshow(
                "Adversarial Patch",
                patch_image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):

                break

    finally:
        summary = metrics.summary()

        print()

        print("=" * 60)
        print("Physical Evaluation Summary")
        print("=" * 60)

        print(
            f"Frames Processed          : {summary['frames']}"
        )

        print(
            f"Average FPS              : {summary['average_fps']:.2f}"
        )

        print(
            f"Average Objects          : {summary['average_objects']:.2f}"
        )

        print(
            f"Maximum Objects          : {summary['max_objects']}"
        )

        print(
            f"Minimum Objects          : {summary['min_objects']}"
        )

        print(
            "Average Person Confidence : "
            f"{summary['average_person_confidence']:.4f}"
        )

        print(
            "Maximum Person Confidence : "
            f"{summary['max_person_confidence']:.4f}"
        )

        print(
            "Minimum Person Confidence : "
            f"{summary['min_person_confidence']:.4f}"
        )

        print("=" * 60)
        webcam.release()

        cv2.destroyAllWindows()

        print("=" * 60)
        print("Physical Attack Demo Closed")
        print("=" * 60)


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()