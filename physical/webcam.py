"""
webcam.py

Webcam utilities.

Responsibilities
----------------
- Initialize webcam
- Capture frames
- Release webcam
- Provide reusable interface for physical experiments

Author:
    Rishab Shetty
"""

import cv2


class Webcam:
    """
    Webcam wrapper for physical experiments.
    """

    def __init__(
        self,
        device=0,
        width=640,
        height=480,
    ):

        self.device = device
        self.width = width
        self.height = height

        self.cap = None

    # -------------------------------------------------
    # Open Webcam
    # -------------------------------------------------

    def open(self):

        print("=" * 60)
        print("Initializing Webcam")
        print("=" * 60)

        # Use DirectShow backend on Windows
        self.cap = cv2.VideoCapture(
            self.device,
            cv2.CAP_DSHOW,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open webcam (Device {self.device})."
            )

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.width,
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.height,
        )

        actual_width = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        actual_height = int(
            self.cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        fps = self.cap.get(
            cv2.CAP_PROP_FPS
        )

        print(f"Device      : {self.device}")
        print(
            f"Resolution  : {actual_width} x {actual_height}"
        )
        print(f"FPS         : {fps:.2f}")

        print("=" * 60)

    # -------------------------------------------------
    # Read Frame
    # -------------------------------------------------

    def read(self):

        if self.cap is None:
            raise RuntimeError(
                "Webcam has not been initialized."
            )

        return self.cap.read()

    # -------------------------------------------------
    # Release Webcam
    # -------------------------------------------------

    def release(self):

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        cv2.destroyAllWindows()

        print("=" * 60)
        print("Webcam Released")
        print("=" * 60)


# =====================================================
# Module Test
# =====================================================

if __name__ == "__main__":

    webcam = Webcam()

    webcam.open()

    print("Press 'Q' to exit.")

    while True:

        ret, frame = webcam.read()

        if not ret:
            print("Failed to capture frame.")
            break

        cv2.imshow(
            "Physical Demo - Webcam",
            frame,
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q"):
            break

    webcam.release()