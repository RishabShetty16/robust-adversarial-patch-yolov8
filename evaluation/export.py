"""
export.py

Export evaluation results.

Author:
    Rishab Shetty
"""

from pathlib import Path
import csv
import json


def export_csv(original, patched, suppression, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow(["Metric", "Original", "Patched"])

        writer.writerow(["Persons",
                         original["count"],
                         patched["count"]])

        writer.writerow(["Average Confidence",
                         f"{original['average']:.4f}",
                         f"{patched['average']:.4f}"])

        writer.writerow(["Maximum Confidence",
                         f"{original['maximum']:.4f}",
                         f"{patched['maximum']:.4f}"])

        writer.writerow(["Minimum Confidence",
                         f"{original['minimum']:.4f}",
                         f"{patched['minimum']:.4f}"])

        writer.writerow(["Suppression Rate (%)",
                         f"{suppression:.2f}",
                         ""])

    print(f"✓ Saved {output_path}")


def export_json(original, patched, suppression, output_path):

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "original": original,
        "patched": patched,
        "suppression_rate": suppression,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✓ Saved {output_path}")