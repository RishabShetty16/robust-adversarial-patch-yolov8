"""
export.py

Physical evaluation exporter.

Responsibilities
----------------
- Export physical evaluation results
- Save CSV
- Save JSON
- Create output directories automatically

Author:
    Rishab Shetty
"""

from pathlib import Path

import csv
import json


class PhysicalExporter:
    """
    Export physical evaluation results.
    """

    def __init__(
        self,
        output_dir="outputs/physical",
    ):

        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.csv_path = (
            self.output_dir
            / "physical_results.csv"
        )

        self.json_path = (
            self.output_dir
            / "physical_results.json"
        )
    # -------------------------------------------------
    # Save JSON
    # -------------------------------------------------

    def save_json(
        self,
        summary,
    ):
        """
        Save evaluation summary as JSON.
        """

        with open(
            self.json_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                summary,
                file,
                indent=4,
            )

        print(
            f"✓ Saved {self.json_path}"
        )

    # -------------------------------------------------
    # Save CSV
    # -------------------------------------------------

    def save_csv(
        self,
        summary,
    ):
        """
        Save evaluation summary as CSV.
        """

        with open(
            self.csv_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:

            writer = csv.writer(file)

            writer.writerow(
                ["Metric", "Value"]
            )

            for key, value in summary.items():

                writer.writerow(
                    [key, value]
                )

        print(
            f"✓ Saved {self.csv_path}"
        )

    # -------------------------------------------------
    # Export Results
    # -------------------------------------------------

    def export(
        self,
        summary,
    ):
        """
        Export physical evaluation results.
        """

        print()

        print("=" * 60)
        print("Exporting Physical Results")
        print("=" * 60)

        self.save_csv(summary)

        self.save_json(summary)

        print("=" * 60)
        print("✓ Physical Results Exported")
        print("=" * 60)