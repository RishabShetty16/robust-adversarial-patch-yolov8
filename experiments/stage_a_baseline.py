"""
Stage A Baseline

Single-image adversarial patch optimization.

Author:
    Rishab Shetty
"""

from pathlib import Path

import torch
import torch.optim as optim

from attack.config import load_config, get_device
from attack.dataset import COCODataset
from attack.patch import AdversarialPatch
from attack.patch_applier import PatchApplier
from attack.detector import YOLODetector
from attack.parser import DetectionParser
from attack.losses import confidence_loss

cfg = load_config("attack/configs/default.yaml")

device = get_device(cfg)

print("=" * 60)
print("Stage A Baseline")
print("=" * 60)
print("Device :", device)
print("=" * 60)

# ==========================================================
# Dataset
# ==========================================================

dataset = COCODataset(cfg)

print(dataset)

image = dataset[0]

print()

print("Image Shape :", image.shape) 

# ==========================================================
# Patch
# ==========================================================

patch = AdversarialPatch(
    size=cfg["patch"]["size"]
)

patch = patch.to(device)

print()

print(patch)

# ==========================================================
# Optimizer
# ==========================================================

optimizer = optim.Adam(
    patch.parameters(),
    lr=cfg["optimizer"]["lr"]
)

print()

print("Optimizer Created")

# ==========================================================
# Patch Applier
# ==========================================================

applier = PatchApplier()

patched = applier.apply(
    image=image,
    patch=patch(),
    x=200,
    y=200,
)

print()

print("Patched Image Shape :", patched.shape)

from torchvision.transforms.functional import to_pil_image

output_path = Path("outputs/figures/patched_image.png")
output_path.parent.mkdir(parents=True, exist_ok=True)

to_pil_image(patched.cpu()).save(output_path)

print("Saved :", output_path)

# ==========================================================
# Detector
# ==========================================================

print()
print("=" * 60)
print("Running YOLO Detection")
print("=" * 60)

detector = YOLODetector(cfg)

results = detector(str(output_path))

print()

print("Number of Result Objects :", len(results))

# ==========================================================
# Parser
# ==========================================================

parser = DetectionParser()

detections = parser.parse(results)

print()

print("Detections Found :", len(detections))

print()

for i, detection in enumerate(detections):

    print(f"Detection {i+1}")

    print("Class      :", detection["class_name"])
    print("Confidence :", round(detection["confidence"], 4))
    print("BBox       :", detection["bbox"])

    print()

confidences = [
    d["confidence"]
    for d in detections
]
print()
print("=" * 60)
print("Experiment Summary")
print("=" * 60)

print(f"Patch Size      : {cfg['patch']['size']}")
print(f"Patch Position  : (200, 200)")
print(f"Detections      : {len(detections)}")
print(f"Max Confidence  : {max(confidences):.4f}")
print(f"Mean Confidence : {sum(confidences)/len(confidences):.4f}")

print("=" * 60)