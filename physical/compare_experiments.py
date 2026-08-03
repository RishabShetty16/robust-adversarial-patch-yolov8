"""
compare_experiments.py

Compare two physical experiments.

Responsibilities
----------------
- Load baseline experiment
- Load patched experiment
- Compare both experiments
- Print comparison summary

Author:
    Rishab Shetty
"""

import json
from pathlib import Path

from physical.comparison import PhysicalComparison


# ==========================================================
# Helper
# ==========================================================

def load_summary(path):

    path = Path(path)

    if not path.exists():

        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ==========================================================
# Main
# ==========================================================

def main():

    print("=" * 60)
    print("Physical Experiment Comparison")
    print("=" * 60)

    baseline_path = input(
        "Baseline JSON : "
    ).strip()

    patched_path = input(
        "Patched JSON  : "
    ).strip()

    baseline = load_summary(
        baseline_path,
    )

    patched = load_summary(
        patched_path,
    )

    comparison = PhysicalComparison()

    comparison.set_baseline(
        baseline,
    )

    comparison.set_patched(
        patched,
    )

    comparison.summary()


# ==========================================================
# Entry Point
# ==========================================================

if __name__ == "__main__":

    main()