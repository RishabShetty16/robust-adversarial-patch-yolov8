# Robust Adversarial Patch Attack for YOLOv8

## Overview

This repository implements a modular and research-oriented framework for developing robust adversarial patch attacks against YOLOv8.

The project is being built incrementally, with each milestone introducing a new component of the adversarial attack pipeline while maintaining a clean and extensible architecture.

The long-term objective is to generate physically robust adversarial patches capable of suppressing object detection under real-world transformations.

---

# Current Pipeline

```
Configuration
      │
      ▼
Dataset Loader
      │
      ▼
Adversarial Patch
      │
      ▼
Patch Applier
      │
      ▼
YOLOv8 Detector
      │
      ▼
Attack Target Extraction
      │
      ▼
Baseline Optimization Loss
      │
      ▼
Backpropagation
      │
      ▼
Optimizer
      │
      ▼
Updated Patch
```

---

# Project Progress

## ✅ Completed

- Configuration Management
- Utility Functions
- Adversarial Patch Module
- Dataset Loader
- Patch Applier
- YOLOv8 Detector Wrapper
- Detection Parser
- Loss Function Framework
- Attack Target Abstraction
- Differentiable YOLO Forward Pass
- End-to-End Optimization Pipeline
- Gradient Verification
- Stage A Baseline Experiment

---

## 🚧 In Progress

- Multi-Epoch Training Loop
- DataLoader Integration
- Checkpoint Saving
- Logging
- Person Suppression Loss
- Expectation over Transformation (EOT)

---

## 📅 Planned

- Physical Attack Pipeline
- Non-Printability Score (NPS)
- Patch Regularization
- Evaluation Metrics
- Benchmark Experiments
- COCO Evaluation
- Physical World Validation

---

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

**Commit 10**

✅ Differentiable Adversarial Patch Optimization Pipeline

Implemented:

- Differentiable YOLO forward pass
- Attack target abstraction
- Baseline optimization loss
- End-to-end gradient propagation
- Optimizer integration
- Verified gradient flow from detector to adversarial patch

---

# Future Milestones

- **Commit 11**
  - Multi-Epoch Training Framework
  - DataLoader
  - Logging
  - Checkpoint Saving

- **Commit 12**
  - Detector-specific Person Suppression Loss
  - First True Adversarial Optimization

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