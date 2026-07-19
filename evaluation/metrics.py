"""
metrics.py

Evaluation metrics for adversarial patch experiments.

Author:
    Rishab Shetty
"""


def extract_confidences(result):
    """
    Extract confidence scores from YOLO detections.

    Parameters
    ----------
    result
        Ultralytics detection result.

    Returns
    -------
    list[float]
    """

    confidences = []

    for box in result.boxes:
        confidences.append(float(box.conf.item()))

    return confidences


def compute_metrics(result):
    """
    Compute basic detection statistics.

    Parameters
    ----------
    result
        Ultralytics detection result.

    Returns
    -------
    dict
    """

    confidences = extract_confidences(result)

    metrics = {
        "count": len(confidences),
        "average": 0.0,
        "maximum": 0.0,
        "minimum": 0.0,
    }

    if confidences:

        metrics["average"] = sum(confidences) / len(confidences)
        metrics["maximum"] = max(confidences)
        metrics["minimum"] = min(confidences)

    return metrics


def compute_suppression(original, patched):
    """
    Compute detection suppression percentage.

    Parameters
    ----------
    original : dict
    patched : dict

    Returns
    -------
    float
    """

    if original["count"] == 0:
        return 0.0

    return (
        (original["count"] - patched["count"])
        / original["count"]
    ) * 100


def print_metrics(title, metrics):
    """
    Pretty print metrics.
    """

    print("=" * 60)
    print(title)
    print("=" * 60)

    print(f"Persons            : {metrics['count']}")
    print(f"Average Confidence : {metrics['average']:.4f}")
    print(f"Maximum Confidence : {metrics['maximum']:.4f}")
    print(f"Minimum Confidence : {metrics['minimum']:.4f}")

    print()