# Robust Adversarial Patch Attack for YOLOv8

## Overview

This repository implements a modular and research-oriented framework for developing robust adversarial patch attacks against YOLOv8.

The project is being built incrementally, with each milestone introducing a new component of the adversarial attack pipeline while maintaining a clean and extensible architecture.

The long-term objective is to generate physically robust adversarial patches capable of suppressing object detection under real-world transformations.

---

# Current Pipeline
---

```
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
                    Patch Applier
                           │
                           ▼
                      YOLO Detector
                           │
                           ▼
                    Attack Target
                           │
                           ▼
                     Baseline Loss
                           │
                           ▼
                     Backpropagation
                           │
                           ▼
                       Optimizer
                           │
                           ▼
                    Updated Patch
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
      Checkpoint Saving          Training Logs
```

---

## Current Progress

### ✅ Completed

- Configuration system
- COCO dataset loader
- Adversarial patch representation
- Patch application
- YOLO detector wrapper
- Detection parser
- Attack target abstraction
- Differentiable optimization pipeline
- Multi-epoch training
- DataLoader integration
- Checkpoint saving
- Loss history logging
- Patch statistics logging

### 🚧 In Progress

- Person suppression objective
- EOT transformations
- Physical robustness
- Evaluation metrics

### 📌 Planned

- Attack Success Rate (ASR)
- Multi-image optimization
- Physical-world evaluation
- Benchmark experiments

# Repository Structure

```
attack/
│
├── configs/
├── attack_target.py
├── config.py
├── dataset.py
├── detector.py
├── losses.py
├── parser.py
├── patch.py
├── patch_applier.py
├── trainer.py
├── utils.py
│
experiments/
│
├── stage_a_baseline.py
├── stage_a_train.py
│
tests/
│
data/
│
outputs/
│
├── checkpoints/
├── figures/
├── logs/
└── patches/
```

---

# Current Milestone

## Commit 12: Detector-Aware Suppression Loss

### Objective

Implemented a detector-aware optimization objective for adversarial patch training.

### Changes

- Added `AttackTarget` module to extract target class confidence scores from YOLOv8 predictions.
- Implemented `person_suppression_loss()` to minimize person detection confidence.
- Replaced the placeholder baseline loss with detector-aware suppression loss.
- Added confidence statistics during training:
  - Mean Target Confidence
  - Maximum Target Confidence
  - Minimum Target Confidence
- Added gradient flow verification.
- Added patch clamping after each optimizer step to keep pixel values within the valid image range `[0,1]`.
- Added checkpoint saving and loss history logging.

### Training Status

Current pipeline:

Input Image
↓
Adversarial Patch
↓
Patch Application
↓
YOLOv8 Detector
↓
Target Score Extraction
↓
Suppression Loss
↓
Backpropagation
↓
Patch Update

# Future Milestones

- **Commit 13**
  - Expectation over Transformation (EOT)

- **Commit 14**
  - Physical Attack Enhancements

---

# Tech Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy

---

# Project Status

**Current Stage:** Stage A — Differentiable Optimization Pipeline ✅

The repository now supports end-to-end differentiable optimization of adversarial patches. Upcoming milestones focus on scalable training, detector-specific attack objectives, and physical robustness.