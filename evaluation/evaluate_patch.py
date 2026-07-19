"""
evaluate_patch.py

Evaluate a trained adversarial patch.

Author:
    Rishab Shetty
"""

import torch
from evaluation.metrics import (
    compute_metrics,
    compute_suppression,
    print_metrics,
)
from evaluation.export import (
    export_csv,
    export_json,
)
from evaluation.comparison import create_comparison
from evaluation.visualization import save_detection_image
from attack.config import load_config
from attack.detector import YOLODetector
from attack.patch import AdversarialPatch
from attack.patch_applier import PatchApplier
from attack.dataset import COCODataset


def main():

    print("=" * 60)
    print("Patch Evaluation")
    print("=" * 60)

    # --------------------------------------------------
    # Load Config
    # --------------------------------------------------

    cfg = load_config(
        "attack/configs/default.yaml"
    )

    print("✓ Configuration Loaded")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = COCODataset(cfg)

    print("✓ Dataset Loaded")

    # --------------------------------------------------
    # Detector
    # --------------------------------------------------

    detector = YOLODetector(cfg)

    print("✓ Detector Loaded")

    # --------------------------------------------------
    # Patch
    # --------------------------------------------------

    patch = AdversarialPatch(
        size=cfg["patch"]["size"],
        initialization=cfg["patch"]["initialization"],
        clamp_min=cfg["patch"]["clamp_min"],
        clamp_max=cfg["patch"]["clamp_max"],
    )

    print("✓ Patch Initialized")

    # --------------------------------------------------
    # Patch Applier
    # --------------------------------------------------

    patch_applier = PatchApplier()

    print("✓ Patch Applier Initialized")

    import glob
    import os

    # --------------------------------------------------
    # Load Latest Checkpoint
    # --------------------------------------------------

    checkpoint_files = sorted(
        glob.glob("outputs/checkpoints/epoch_*.pt")
    )

    if not checkpoint_files:
        raise FileNotFoundError("No checkpoints found.")

    checkpoint_path = checkpoint_files[-1]

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    patch.load_state_dict(checkpoint["patch"])

    print("✓ Checkpoint Loaded")
    print("Checkpoint File :", os.path.basename(checkpoint_path))
    print(f"Checkpoint Epoch : {checkpoint['epoch']}")
    # ==================================================
    # Load Evaluation Image
    # ==================================================

    print()
    print("=" * 60)
    print("Loading Evaluation Image")
    print("=" * 60)

    image = dataset[0]

    print("✓ Image Loaded")
    print("Image Shape :", image.shape)

    image = image.unsqueeze(0)

    print("Batch Shape :", image.shape)

    # ==================================================
    # YOLO Detection (Original Image)
    # ==================================================

    print()
    print("=" * 60)
    print("Running YOLO on Original Image")
    print("=" * 60)

    results = detector.predict(image)

    print("✓ Detection Completed")

    result = results[0]

    print()
    print("=" * 60)
    print("Original Detection Summary")
    print("=" * 60)

    print("Boxes Found :", len(result.boxes))

    if len(result.boxes) == 0:

        print("No detections found.")

    else:

        for i, box in enumerate(result.boxes):

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            print(
                f"Detection {i + 1}: "
                f"Class={cls}, "
                f"Confidence={conf:.4f}"
            )

    # ==================================================
    # Apply Trained Patch
    # ==================================================

    print()
    print("=" * 60)
    print("Applying Trained Patch")
    print("=" * 60)

    patched_image = patch_applier.apply(
        image=image,
        patch=patch(),
        x=100,
        y=100,
    )

    print("✓ Patch Applied")
    print("Patched Image Shape :", patched_image.shape)

    # ==================================================
    # YOLO Detection (Patched Image)
    # ==================================================

    print()
    print("=" * 60)
    print("Running YOLO on Patched Image")
    print("=" * 60)

    patched_results = detector.predict(
        patched_image
    )

    print("✓ Detection Completed")

    patched_result = patched_results[0]

    print()
    print("=" * 60)
    print("Patched Detection Summary")
    print("=" * 60)

    print("Boxes Found :", len(patched_result.boxes))

    if len(patched_result.boxes) == 0:

        print("No detections found.")

    else:

        for i, box in enumerate(patched_result.boxes):

            cls = int(box.cls.item())
            conf = float(box.conf.item())

            print(
                f"Detection {i + 1}: "
                f"Class={cls}, "
                f"Confidence={conf:.4f}"
            )

    # ==================================================
    # Metrics
    # ==================================================

    original_metrics = compute_metrics(result)
    patched_metrics = compute_metrics(patched_result)

    suppression = compute_suppression(
        original_metrics,
        patched_metrics,
    )

    print()
    print_metrics(
        "Original Metrics",
        original_metrics,
    )

    print_metrics(
        "Patched Metrics",
        patched_metrics,
    )

    print("=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Suppression Rate : {suppression:.2f}%")

    # ==================================================
    # Save Detection Images
    # ==================================================

    print()
    print("=" * 60)
    print("Saving Detection Images")
    print("=" * 60)

    save_detection_image(
        result,
        "outputs/evaluation/original_detection.jpg",
    )

    save_detection_image(
        patched_result,
        "outputs/evaluation/patched_detection.jpg",
    )

    # ==================================================
    # Create Comparison Image
    # ==================================================

    print()
    print("=" * 60)
    print("Creating Comparison Image")
    print("=" * 60)

    create_comparison(
        "outputs/evaluation/original_detection.jpg",
        "outputs/evaluation/patched_detection.jpg",
        "outputs/evaluation/comparison.jpg",
    )

    # ==================================================
    # Export Results
    # ==================================================

    print()
    print("=" * 60)
    print("Exporting Results")
    print("=" * 60)

    export_csv(
        original_metrics,
        patched_metrics,
        suppression,
        "outputs/evaluation/results.csv",
    )

    export_json(
        original_metrics,
        patched_metrics,
        suppression,
        "outputs/evaluation/results.json",
    )

    print()
    print("✓ Evaluation Complete")


if __name__ == "__main__":
    main()