# Robust Adversarial Patch Attack for YOLOv8

A modular, research-oriented framework for developing **robust adversarial patch attacks** against **YOLOv8** using PyTorch.

The goal of this project is to generate **universal adversarial patches** capable of suppressing object detection while remaining effective under various transformations, forming the foundation for future physical-world adversarial attacks.

---

# Features

- Modular adversarial patch implementation
- YOLOv8 detector integration
- COCO dataset support
- End-to-end differentiable optimization
- Person suppression objective
- Random patch placement
- Expectation over Transformation (EOT)
- Configurable patch initialization
- Cosine learning rate scheduler
- Multi-epoch training pipeline
- Automatic checkpoint saving
- Training loss logging
- Comprehensive evaluation framework
- CSV and JSON metric export
- Detection visualization

---

# Current Training Pipeline

```text
                     Configuration
                           │
                           ▼
                 COCO Dataset Loader
                           │
                           ▼
                    PyTorch DataLoader
                           │
                           ▼
                 Adversarial Patch
                           │
                           ▼
               Expectation over Transformation
                           │
                           ▼
                  Random Patch Placement
                           │
                           ▼
                     Patch Applier
                           │
                           ▼
                     YOLOv8 Detector
                           │
                           ▼
                 Detection Parsing
                           │
                           ▼
                 Attack Target Selection
                           │
                           ▼
             Person Suppression Loss
                           │
                           ▼
                    Backpropagation
                           │
                           ▼
                      Adam Optimizer
                           │
                           ▼
              Cosine Learning Rate Scheduler
                           │
                           ▼
                    Updated Patch
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
   Checkpoint Saving              Loss History Logging
```

---

# Repository Structure

```text
robust-adversarial-patch-yolov8/

│
├── attack/
│   ├── configs/
│   │   └── default.yaml
│   ├── attack_target.py
│   ├── config.py
│   ├── dataset.py
│   ├── detector.py
│   ├── eot.py
│   ├── losses.py
│   ├── parser.py
│   ├── patch.py
│   ├── patch_applier.py
│   ├── trainer.py
│   └── utils.py
│
├── evaluation/
│   ├── evaluate_patch.py
│   ├── export.py
│   ├── metrics.py
│   └── visualization.py
│
├── experiments/
│
├── tests/
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── patches/
│
├── data/
│
├── evaluate.py
├── train.py
├── requirements.txt
└── README.md
```

---

# Implemented Components

## Adversarial Patch

- Learnable universal adversarial patch
- Multiple initialization strategies
  - Gray
  - Random
  - Checkerboard
  - Gaussian
- Pixel value clamping
- Automatic checkpoint serialization

---

## Dataset

- COCO image loader
- PyTorch Dataset interface
- DataLoader integration
- Configurable batch size
- Configurable image resolution

---

## Detector

- Ultralytics YOLOv8 integration
- Configurable confidence threshold
- Configurable IoU threshold
- Device auto-selection
- Detection parsing

---

## Attack Target

Supports detector-aware optimization by extracting:

- Person confidence scores
- Top-K detections
- Batch-wise target tensors

---

## Loss Function

Current objective:

- Person Suppression Loss

The optimization minimizes person detection confidence while maintaining gradient flow through the detector.

---

## Expectation over Transformation (EOT)

Current transformations include:

- Rotation
- Scaling

The framework is designed for additional physical-world transformations.

---

## Training Engine

The trainer supports:

- Multi-epoch optimization
- Random patch placement
- Automatic gradient computation
- Adam optimizer
- Cosine learning rate scheduling
- Epoch logging
- Patch statistics
- Automatic checkpoint saving
- CSV loss logging

---

# Evaluation Framework

The repository includes a complete evaluation pipeline.

Features:

- Original image inference
- Patched image inference
- Detection comparison
- Suppression metrics
- Confidence statistics
- Annotated visualizations
- Side-by-side comparison images
- CSV export
- JSON export

Example output:

```text
Original Persons : 3
Patched Persons  : 2

Suppression Rate : 33.33%

Confidence Drop  : 21.48%

Retention Rate   : 66.67%
```

---

# Training

Run training:

```bash
python train.py
```

---

# Evaluation

Run evaluation:

```bash
python -m evaluation.evaluate_patch
```

Generated outputs:

```text
outputs/

├── checkpoints/
├── figures/
│   ├── original_detection.jpg
│   ├── patched_detection.jpg
│   └── comparison.jpg
│
├── logs/
│   ├── loss_history.csv
│   ├── results.csv
│   └── results.json
│
└── patches/
```

---

# Current Progress

| Component | Status |
|-----------|--------|
| Configuration System | ✅ |
| COCO Dataset Loader | ✅ |
| YOLOv8 Integration | ✅ |
| Detection Parser | ✅ |
| Adversarial Patch | ✅ |
| Patch Application | ✅ |
| Random Patch Placement | ✅ |
| Attack Target Extraction | ✅ |
| Person Suppression Loss | ✅ |
| EOT (Rotation + Scaling) | ✅ |
| Multi-Epoch Training | ✅ |
| Checkpoint Saving | ✅ |
| Loss Logging | ✅ |
| Patch Initialization Strategies | ✅ |
| Cosine LR Scheduler | ✅ |
| Evaluation Framework | ✅ |
| Visualization | ✅ |
| CSV / JSON Export | ✅ |

---

# Roadmap

Upcoming improvements include:

- Best checkpoint selection
- Training loss visualization
- Multi-image optimization
- Multi-image evaluation
- Stronger EOT transformations
- Physical-world adversarial evaluation
- Patch robustness benchmarking
- Attack Success Rate (ASR)
- Detector transferability experiments

---

# Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PyYAML

---

# Project Status

**Current Stage:** Research Prototype (Commit 20)

The repository now implements a complete end-to-end differentiable adversarial patch optimization pipeline with configurable initialization strategies, expectation over transformation, cosine learning rate scheduling, and a comprehensive evaluation framework.

Future work will focus on improving attack robustness, scaling training to larger datasets, and evaluating physical-world performance.

---

# Author

**Rishab Shetty**

Computer Science (AI & ML)

PES University

Research Interests:

- Adversarial Machine Learning
- Computer Vision
- Deep Learning
- AI Security