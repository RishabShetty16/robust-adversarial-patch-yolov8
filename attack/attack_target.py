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

    def __init__(self, cfg):

        self.cfg = cfg
        self.target_class = cfg["attack"]["target_class"]

    def extract(self, outputs):
        """
        Extract optimization targets from detector outputs.
        """

        predictions = outputs[0]

        return {
            "predictions": predictions
        }

    def __repr__(self):

        return (
            f"AttackTarget(target_class='{self.target_class}')"
        )