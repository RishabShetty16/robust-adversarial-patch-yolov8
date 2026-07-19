"""
metrics.py

Evaluation metrics for adversarial patch experiments.

Author:
    Rishab Shetty
"""

PERSON_CLASS = 0


# ---------------------------------------------------------
# Extract Person Confidences
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Compute Metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Detection Suppression
# ---------------------------------------------------------

def compute_suppression(original, patched):
    """
    Percentage reduction in detected persons.
    """

    if original["count"] == 0:
        return 0.0

    suppression = (
        (original["count"] - patched["count"])
        / original["count"]
    ) * 100

    return max(0.0, suppression)


# ---------------------------------------------------------
# Confidence Drop
# ---------------------------------------------------------

def compute_confidence_drop(original, patched):
    """
    Percentage reduction in average confidence.
    """

    if original["average"] == 0:
        return 0.0

    drop = (
        (original["average"] - patched["average"])
        / original["average"]
    ) * 100

    return max(0.0, drop)


# ---------------------------------------------------------
# Detection Retention
# ---------------------------------------------------------

def compute_retention(original, patched):
    """
    Percentage of person detections that remain.
    """

    if original["count"] == 0:
        return 0.0

    retention = (
        patched["count"]
        / original["count"]
    ) * 100

    return min(100.0, retention)


# ---------------------------------------------------------
# Print Metrics
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Print Evaluation Summary
# ---------------------------------------------------------

def print_summary(
    original,
    patched,
    suppression,
    confidence_drop,
    retention,
):
    """
    Print a concise evaluation summary.
    """

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    print(f"Persons Before          : {original['count']}")
    print(f"Persons After           : {patched['count']}")
    print()

    print(f"Average Confidence Before : {original['average']:.4f}")
    print(f"Average Confidence After  : {patched['average']:.4f}")
    print()

    print(f"Detection Suppression   : {suppression:.2f}%")
    print(f"Confidence Drop         : {confidence_drop:.2f}%")
    print(f"Detection Retention     : {retention:.2f}%")

    print("=" * 60)