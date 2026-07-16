"""
parser.py

YOLO Detection Parser

Responsibilities
----------------
Convert raw Ultralytics YOLO results into a clean,
detector-independent format.

Author:
    Rishab Shetty
"""

from __future__ import annotations


class DetectionParser:
    """
    Parse YOLO detections into a standard format.
    """

    def __init__(self):
        pass

    # -----------------------------------------------------
    # Parse predictions
    # -----------------------------------------------------

    def parse(self, results):

        detections = []

        if len(results) == 0:
            return detections

        result = results[0]

        boxes = result.boxes

        if boxes is None:

            return detections

        names = result.names

        for box in boxes:

            detections.append({

                "class_id": int(box.cls.item()),

                "class_name": names[int(box.cls.item())],

                "confidence": float(box.conf.item()),

                "bbox": box.xyxy.squeeze().tolist(),

            })

        return detections

    # -----------------------------------------------------

    def __repr__(self):

        return "DetectionParser()"