"""
comparison.py

Physical experiment comparison.

Responsibilities
----------------
- Compare baseline and patched experiments
- Compute physical attack metrics
- Return comparison summary

Author:
    Rishab Shetty
"""


class PhysicalComparison:
    """
    Compare two physical evaluation summaries.
    """

    def __init__(self):

        self.baseline = None

        self.patched = None

    # -------------------------------------------------
    # Set Baseline
    # -------------------------------------------------

    def set_baseline(
        self,
        summary,
    ):

        self.baseline = summary

    # -------------------------------------------------
    # Set Patched
    # -------------------------------------------------

    def set_patched(
        self,
        summary,
    ):

        self.patched = summary

    # -------------------------------------------------
    # Compare Experiments
    # -------------------------------------------------

    def compare(self):
        """
        Compare baseline and patched experiments.
        """

        if self.baseline is None:
            raise RuntimeError(
                "Baseline experiment not set."
            )

        if self.patched is None:
            raise RuntimeError(
                "Patched experiment not set."
            )

        baseline = self.baseline

        patched = self.patched

        # ---------------------------------------------
        # Average Objects
        # ---------------------------------------------

        baseline_objects = baseline["average_objects"]
        patched_objects = patched["average_objects"]

        if baseline_objects > 0:

            object_drop = (
                (baseline_objects - patched_objects)
                / baseline_objects
            ) * 100.0

        else:

            object_drop = 0.0

        # ---------------------------------------------
        # Average Persons
        # ---------------------------------------------

        baseline_persons = baseline["average_persons"]
        patched_persons = patched["average_persons"]

        if baseline_persons > 0:

            person_drop = (
                (baseline_persons - patched_persons)
                / baseline_persons
            ) * 100.0

        else:

            person_drop = 0.0

        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        baseline_confidence = (
            baseline["average_person_confidence"]
        )

        patched_confidence = (
            patched["average_person_confidence"]
        )

        if baseline_confidence > 0:

            confidence_drop = (
                (
                    baseline_confidence
                    - patched_confidence
                )
                / baseline_confidence
            ) * 100.0

        else:

            confidence_drop = 0.0

        # ---------------------------------------------
        # Physical Suppression Rate
        # ---------------------------------------------

        suppression_rate = max(
            person_drop,
            0.0,
        )

        return {

            "baseline_objects": baseline_objects,

            "patched_objects": patched_objects,

            "object_drop_percent": object_drop,

            "baseline_persons": baseline_persons,

            "patched_persons": patched_persons,

            "person_drop_percent": person_drop,

            "baseline_person_confidence":
                baseline_confidence,

            "patched_person_confidence":
                patched_confidence,

            "confidence_drop_percent":
                confidence_drop,

            "physical_suppression_rate":
                suppression_rate,

        }

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    def summary(self):
        """
        Return comparison summary.
        """

        comparison = self.compare()

        print()

        print("=" * 60)
        print("Physical Experiment Comparison")
        print("=" * 60)

        print()

        print(
            f"Baseline Objects           : "
            f"{comparison['baseline_objects']:.2f}"
        )

        print(
            f"Patched Objects            : "
            f"{comparison['patched_objects']:.2f}"
        )

        print(
            f"Object Drop (%)            : "
            f"{comparison['object_drop_percent']:.2f}"
        )

        print()

        print(
            f"Baseline Persons           : "
            f"{comparison['baseline_persons']:.2f}"
        )

        print(
            f"Patched Persons            : "
            f"{comparison['patched_persons']:.2f}"
        )

        print(
            f"Person Drop (%)            : "
            f"{comparison['person_drop_percent']:.2f}"
        )

        print()

        print(
            f"Baseline Confidence        : "
            f"{comparison['baseline_person_confidence']:.4f}"
        )

        print(
            f"Patched Confidence         : "
            f"{comparison['patched_person_confidence']:.4f}"
        )

        print(
            f"Confidence Drop (%)        : "
            f"{comparison['confidence_drop_percent']:.2f}"
        )

        print()

        print(
            f"Physical Suppression Rate  : "
            f"{comparison['physical_suppression_rate']:.2f}%"
        )

        print("=" * 60)

        return comparison