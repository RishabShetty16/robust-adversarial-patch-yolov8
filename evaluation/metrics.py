"""
metrics.py

Evaluation metrics for adversarial patch experiments.

Author:
    Rishab Shetty
"""

PERSON_CLASS = 0


def extract_person_confidences(result):
    """
    Extract confidence scores for PERSON detections only.
    """

    confidences = []

    for box in result.boxes:

        class_id = int(box.cls.item())

        if class_id != PERSON_CLASS:
            continue

        confidences.append(float(box.conf.item()))

    return confidences


def compute_metrics(result):
    """
    Compute statistics for PERSON detections only.
    """

    confidences = extract_person_confidences(result)

    metrics = {
        "count": len(confidences),
        "average": 0.0,
        "maximum": 0.0,
        "minimum": 0.0,
    }

    if len(confidences) > 0:

        metrics["average"] = sum(confidences) / len(confidences)
        metrics["maximum"] = max(confidences)
        metrics["minimum"] = min(confidences)

    return metrics


def compute_suppression(original, patched):
    """
    Compute person detection suppression percentage.
    """

    if original["count"] == 0:
        return 0.0

    return (
        (original["count"] - patched["count"])
        / original["count"]
    ) * 100


def print_metrics(title, metrics):
    """
    Pretty-print person detection statistics.
    """

    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"Person Detections  : {metrics['count']}")
    print(f"Average Confidence : {metrics['average']:.4f}")
    print(f"Maximum Confidence : {metrics['maximum']:.4f}")
    print(f"Minimum Confidence : {metrics['minimum']:.4f}")

    print()