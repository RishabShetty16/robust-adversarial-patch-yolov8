"""
export.py

Export evaluation results.

Author:
    Rishab Shetty
"""

from pathlib import Path
import csv
import json


# ---------------------------------------------------------
# Export CSV
# ---------------------------------------------------------

def export_csv(
    original,
    patched,
    suppression,
    confidence_drop,
    retention,
    output_path,
):
    """
    Export evaluation metrics to CSV.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Value"])

        writer.writerow(["Persons Before", original["count"]])
        writer.writerow(["Persons After", patched["count"]])

        writer.writerow(
            [
                "Average Confidence Before",
                f"{original['average']:.4f}",
            ]
        )

        writer.writerow(
            [
                "Average Confidence After",
                f"{patched['average']:.4f}",
            ]
        )

        writer.writerow(
            [
                "Maximum Confidence Before",
                f"{original['maximum']:.4f}",
            ]
        )

        writer.writerow(
            [
                "Maximum Confidence After",
                f"{patched['maximum']:.4f}",
            ]
        )

        writer.writerow(
            [
                "Minimum Confidence Before",
                f"{original['minimum']:.4f}",
            ]
        )

        writer.writerow(
            [
                "Minimum Confidence After",
                f"{patched['minimum']:.4f}",
            ]
        )

        writer.writerow(
            [
                "Detection Suppression (%)",
                f"{suppression:.2f}",
            ]
        )

        writer.writerow(
            [
                "Confidence Drop (%)",
                f"{confidence_drop:.2f}",
            ]
        )

        writer.writerow(
            [
                "Detection Retention (%)",
                f"{retention:.2f}",
            ]
        )

    print(f"✓ Saved {output_path}")


# ---------------------------------------------------------
# Export JSON
# ---------------------------------------------------------

def export_json(
    original,
    patched,
    suppression,
    confidence_drop,
    retention,
    output_path,
):
    """
    Export evaluation metrics to JSON.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "original": original,
        "patched": patched,
        "suppression_rate": suppression,
        "confidence_drop": confidence_drop,
        "retention_rate": retention,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✓ Saved {output_path}")