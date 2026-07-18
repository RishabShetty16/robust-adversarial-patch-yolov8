"""
attack_target.py

Extract optimization targets from detector outputs.

Responsibilities
----------------
- Interpret detector outputs
- Extract attack objectives
- Isolate detector-specific logic

Author:
    Rishab Shetty
"""


class AttackTarget:
    """
    Converts detector outputs into attack targets.
    """

    # COCO class mapping
    COCO_CLASS_IDS = {
        "person": 0,
        "bicycle": 1,
        "car": 2,
        "motorcycle": 3,
        "airplane": 4,
        "bus": 5,
        "train": 6,
        "truck": 7,
        "boat": 8,
    }

    def __init__(self, cfg):

        self.cfg = cfg

        target = cfg["attack"]["target_class"]

        # Allow either:
        # target_class: person
        # or
        # target_class: 0
        if isinstance(target, str):
            if target not in self.COCO_CLASS_IDS:
                raise ValueError(
                    f"Unknown target class '{target}'."
                )

            self.target_class = self.COCO_CLASS_IDS[target]

        else:
            self.target_class = int(target)

    def extract(self, outputs):
        """
        Extract optimization targets from detector outputs.

        YOLOv8 Output
        ------------
        predictions : (B, 84, N)

        Channels
        --------
        0:4   -> Bounding boxes
        4:84  -> Class scores
        """

        predictions = outputs[0]

        # Bounding boxes
        boxes = predictions[:, :4, :]

        # All class scores
        class_scores = predictions[:, 4:, :]

        # Target class scores
        target_scores = class_scores[:, self.target_class, :]

        return {
            "predictions": predictions,
            "boxes": boxes,
            "class_scores": class_scores,
            "target_scores": target_scores,
            "person_scores": target_scores,   # backward compatibility
        }

    def __repr__(self):

        return (
            f"AttackTarget(target_class={self.target_class})"
        )