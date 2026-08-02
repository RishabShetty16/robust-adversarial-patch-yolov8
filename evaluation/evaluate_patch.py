"""
evaluate_patch.py

Evaluate a trained adversarial patch on multiple images.

Author:
    Rishab Shetty
"""

import glob
import os

import torch
import numpy as np

from attack.config import load_config
from attack.dataset import COCODataset
from attack.detector import YOLODetector
from attack.patch import AdversarialPatch
from attack.patch_applier import PatchApplier
from attack.placement import get_patch_position
from attack.placement import get_patch_position

from attack.bbox_utils import (
    extract_person_boxes,
    largest_person_box,
)

from evaluation.metrics import (
    compute_metrics,
    compute_suppression,
    compute_confidence_drop,
    compute_retention,
    print_metrics,
    print_summary,
)

from evaluation.export import (
    export_csv,
    export_json,
)

from evaluation.visualization import (
    save_detection_image,
)

from evaluation.comparison import (
    create_comparison,
)


# ============================================================
# Helper Functions
# ============================================================

def load_best_checkpoint(patch):

    best_checkpoint = "outputs/checkpoints/best.pt"

    if os.path.exists(best_checkpoint):

        checkpoint_path = best_checkpoint

        print("✓ Best Checkpoint Found")

    else:

        checkpoint_files = sorted(
            glob.glob("outputs/checkpoints/epoch_*.pt")
        )

        if not checkpoint_files:
            raise FileNotFoundError(
                "No checkpoints found."
            )

        checkpoint_path = checkpoint_files[-1]

        print("Best checkpoint not found.")
        print("Using latest epoch checkpoint.")

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    patch.load_state_dict(
        checkpoint["patch"]
    )

    print()
    print("=" * 60)
    print("Checkpoint Loaded")
    print("=" * 60)
    print("File :", os.path.basename(checkpoint_path))
    print(f"Epoch : {checkpoint['epoch']}")

    if "best_loss" in checkpoint:
        print(
            f"Best Loss : {checkpoint['best_loss']:.6f}"
        )

    print("=" * 60)

    return checkpoint


# ============================================================
# Evaluate Single Image
# ============================================================

def evaluate_single_image(
    image,
    detector,
    patch,
    patch_applier,
    cfg,
):

    image = image.unsqueeze(0)
    # --------------------------------------------
    # Original Detection
    # --------------------------------------------

    original_results = detector.predict(image)

    original_result = original_results[0]

    # --------------------------------------------
    # Detect Persons
    # --------------------------------------------

    person_boxes = extract_person_boxes(
        original_result
    )

    largest_box = largest_person_box(
        person_boxes
    )

    print("Largest Person Box :", largest_box)

    # --------------------------------------------
    # Compute Patch Position
    # --------------------------------------------

    _, _, image_h, image_w = image.shape

    patch_size = patch().shape[-1]

    x, y = get_patch_position(
        image_height=image_h,
        image_width=image_w,
        patch_size=patch_size,
        placement_cfg=cfg["placement"],
        person_box=largest_box,
    )

    print(f"Patch Position : ({x}, {y})")

    # --------------------------------------------
    # Apply Patch
    # --------------------------------------------

    patched_image = patch_applier.apply(
        image=image,
        patch=patch(),
        x=x,
        y=y,
    )
    # --------------------------------------------
    # Patched Detection
    # --------------------------------------------

    patched_results = detector.predict(
        patched_image
    )

    patched_result = patched_results[0]

    # --------------------------------------------
    # Metrics
    # --------------------------------------------

    original_metrics = compute_metrics(
        original_result
    )

    patched_metrics = compute_metrics(
        patched_result
    )

    suppression = compute_suppression(
        original_metrics,
        patched_metrics,
    )

    confidence_drop = compute_confidence_drop(
        original_metrics,
        patched_metrics,
    )

    retention = compute_retention(
        original_metrics,
        patched_metrics,
    )

    return {
        "original_result": original_result,
        "patched_result": patched_result,
        "original_metrics": original_metrics,
        "patched_metrics": patched_metrics,
        "suppression": suppression,
        "confidence_drop": confidence_drop,
        "retention": retention,
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Patch Evaluation")
    print("=" * 60)

    cfg = load_config(
        "attack/configs/default.yaml"
    )

    evaluation_cfg = cfg["evaluation"]

    max_images = evaluation_cfg["max_images"]
    save_visualizations = evaluation_cfg[
        "save_visualizations"
    ]

    print("✓ Configuration Loaded")

    print("=" * 60)
    print("Evaluation Configuration")
    print("=" * 60)
    print("Maximum Images :", max_images)
    print(
        "Save Visualizations :",
        save_visualizations,
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = COCODataset(cfg)

    print("✓ Dataset Loaded")

    num_images = min(
        max_images,
        len(dataset),
    )

    print("=" * 60)
    print("Evaluation Dataset")
    print("=" * 60)
    print("Images Available :", len(dataset))
    print("Images To Evaluate :", num_images)
    print("=" * 60)

    # --------------------------------------------------------
    # Detector
    # --------------------------------------------------------

    detector = YOLODetector(cfg)

    print("✓ Detector Loaded")

    # --------------------------------------------------------
    # Patch
    # --------------------------------------------------------

    patch = AdversarialPatch(
        size=cfg["patch"]["size"],
        initialization=cfg["patch"]["initialization"],
        clamp_min=cfg["patch"]["clamp_min"],
        clamp_max=cfg["patch"]["clamp_max"],
    )

    print("✓ Patch Initialized")

    # --------------------------------------------------------
    # Patch Applier
    # --------------------------------------------------------

    patch_applier = PatchApplier()

    print("✓ Patch Applier Initialized")

    load_best_checkpoint(patch)

    # --------------------------------------------------------
    # Dataset Statistics
    # --------------------------------------------------------

    suppression_scores = []
    confidence_scores = []
    retention_scores = []

    success_flags = []

    first_result = None

    # ========================================================
    # Evaluate Dataset
    # ========================================================

    for index in range(num_images):

        print()
        print("=" * 60)
        print(
            f"Evaluating Image {index + 1}/{num_images}"
        )
        print("=" * 60)

        image = dataset[index]

        evaluation = evaluate_single_image(
            image=image,
            detector=detector,
            patch=patch,
            patch_applier=patch_applier,
            cfg=cfg,
        )

        suppression_scores.append(
            evaluation["suppression"]
        )

        confidence_scores.append(
            evaluation["confidence_drop"]
        )

        retention_scores.append(
            evaluation["retention"]
        )

        success_flags.append(
            evaluation["patched_metrics"]["count"]
            <
            evaluation["original_metrics"]["count"]
        )

        if index == 0:

            first_result = evaluation

            print_metrics(
                "Original Metrics",
                evaluation["original_metrics"],
            )

            print_metrics(
                "Patched Metrics",
                evaluation["patched_metrics"],
            )

            print_summary(
                evaluation["original_metrics"],
                evaluation["patched_metrics"],
                evaluation["suppression"],
                evaluation["confidence_drop"],
                evaluation["retention"],
            )

    # ========================================================
    # Dataset Summary
    # ========================================================

    average_suppression = float(
        np.mean(suppression_scores)
    )

    average_confidence_drop = float(
        np.mean(confidence_scores)
    )

    average_retention = float(
        np.mean(retention_scores)
    )

    attack_success_rate = (
        float(np.mean(success_flags)) * 100.0
    )

    print()

    print("=" * 60)
    print("Dataset Evaluation Summary")
    print("=" * 60)

    print(
        f"Images Evaluated : {num_images}"
    )

    print(
        f"Average Suppression : "
        f"{average_suppression:.2f}%"
    )

    print(
        f"Average Confidence Drop : "
        f"{average_confidence_drop:.2f}%"
    )

    print(
        f"Average Retention : "
        f"{average_retention:.2f}%"
    )

    print(
        f"Attack Success Rate : "
        f"{attack_success_rate:.2f}%"
    )

    print("=" * 60)

    # ========================================================
    # Save Example Visualization
    # ========================================================

    if save_visualizations and first_result is not None:

        print()
        print("=" * 60)
        print("Saving Example Visualizations")
        print("=" * 60)

        save_detection_image(
            first_result["original_result"],
            "outputs/evaluation/original_detection.jpg",
        )

        save_detection_image(
            first_result["patched_result"],
            "outputs/evaluation/patched_detection.jpg",
        )

        create_comparison(
            "outputs/evaluation/original_detection.jpg",
            "outputs/evaluation/patched_detection.jpg",
            "outputs/evaluation/comparison.jpg",
        )

    # ========================================================
    # Export Example Metrics
    # ========================================================

    if first_result is not None:

        print()
        print("=" * 60)
        print("Exporting Results")
        print("=" * 60)

        export_csv(
            first_result["original_metrics"],
            first_result["patched_metrics"],
            first_result["suppression"],
            first_result["confidence_drop"],
            first_result["retention"],
            "outputs/evaluation/results.csv",
        )

        export_json(
            first_result["original_metrics"],
            first_result["patched_metrics"],
            first_result["suppression"],
            first_result["confidence_drop"],
            first_result["retention"],
            "outputs/evaluation/results.json",
        )

    print()

    print("=" * 60)
    print("✓ Evaluation Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()