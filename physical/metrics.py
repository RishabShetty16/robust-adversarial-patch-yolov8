"""
metrics.py

Physical evaluation metrics.

Responsibilities
----------------
- Track live evaluation statistics
- Compute summary metrics
- Reset collected statistics

Author:
    Rishab Shetty
"""


class PhysicalMetrics:
    """
    Collects statistics during
    physical attack experiments.
    """

    def __init__(self):

        self.reset()

    # -------------------------------------------------
    # Reset Statistics
    # -------------------------------------------------

    def reset(self):

        # ---------------------------------------------
        # General Statistics
        # ---------------------------------------------

        self.frames = 0

        self.total_fps = 0.0

        self.total_objects = 0

        self.max_objects = 0

        self.min_objects = float("inf")

        # ---------------------------------------------
        # Person Statistics
        # ---------------------------------------------

        self.total_persons = 0

        self.max_persons = 0

        self.min_persons = float("inf")

        # ---------------------------------------------
        # Person Confidence
        # ---------------------------------------------

        self.total_person_confidence = 0.0

        self.max_person_confidence = 0.0

        self.min_person_confidence = float("inf")

        self.person_detections = 0

    # -------------------------------------------------
    # Update Statistics
    # -------------------------------------------------

    def update(
        self,
        results,
        fps,
    ):
        """
        Update physical evaluation statistics.
        """

        self.frames += 1

        self.total_fps += fps

        boxes = results[0].boxes

        num_objects = len(boxes)

        self.total_objects += num_objects

        self.max_objects = max(
            self.max_objects,
            num_objects,
        )

        self.min_objects = min(
            self.min_objects,
            num_objects,
        )

        # ---------------------------------------------
        # Person Statistics
        # ---------------------------------------------

        person_count = 0

        for box in boxes:

            if int(box.cls) != 0:
                continue

            person_count += 1

            confidence = float(box.conf)

            self.total_person_confidence += confidence

            self.person_detections += 1

            self.max_person_confidence = max(
                self.max_person_confidence,
                confidence,
            )

            self.min_person_confidence = min(
                self.min_person_confidence,
                confidence,
            )

        self.total_persons += person_count

        self.max_persons = max(
            self.max_persons,
            person_count,
        )

        self.min_persons = min(
            self.min_persons,
            person_count,
        )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(self):
        """
        Return summary statistics.
        """

        if self.frames == 0:

            return {

                "frames": 0,

                "average_fps": 0.0,

                "average_objects": 0.0,

                "max_objects": 0,

                "min_objects": 0,

                "average_persons": 0.0,

                "max_persons": 0,

                "min_persons": 0,

                "average_person_confidence": 0.0,

                "max_person_confidence": 0.0,

                "min_person_confidence": 0.0,

            }

        average_fps = (
            self.total_fps
            / self.frames
        )

        average_objects = (
            self.total_objects
            / self.frames
        )

        average_persons = (
            self.total_persons
            / self.frames
        )

        if self.person_detections > 0:

            average_person_confidence = (
                self.total_person_confidence
                / self.person_detections
            )

            min_person_confidence = (
                self.min_person_confidence
            )

        else:

            average_person_confidence = 0.0

            min_person_confidence = 0.0

        if self.min_objects == float("inf"):
            min_objects = 0
        else:
            min_objects = self.min_objects

        if self.min_persons == float("inf"):
            min_persons = 0
        else:
            min_persons = self.min_persons

        return {

            "frames": self.frames,

            "average_fps": average_fps,

            "average_objects": average_objects,

            "max_objects": self.max_objects,

            "min_objects": min_objects,

            "average_persons": average_persons,

            "max_persons": self.max_persons,

            "min_persons": min_persons,

            "average_person_confidence": average_person_confidence,

            "max_person_confidence": self.max_person_confidence,

            "min_person_confidence": min_person_confidence,

        }