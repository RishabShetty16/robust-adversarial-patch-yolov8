# Project Architecture

This document describes the software architecture of the **Robust Adversarial Patch Attack for YOLOv8** framework.

The project follows a **modular, research-oriented architecture**, where each component has a single responsibility. The design emphasizes reproducibility, extensibility, and maintainability, allowing individual modules to evolve independently while supporting end-to-end differentiable adversarial patch optimization.

---

# Architecture Overview

The current implementation provides a complete training and evaluation pipeline for learning universal adversarial patches against YOLOv8.

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
         Expectation over Transformation (EOT)
                           │
                           ▼
              Random Patch Placement
                           │
                           ▼
                    Patch Applier
                           │
                           ▼
                  Patched Image
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
           ┌───────────────┴───────────────┐
           ▼                               ▼
   Checkpoint Saving              Loss History Logging
```

---

# High-Level Design

```text
Configuration
      │
      ▼
Dataset Loader
      │
      ▼
Input Image
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
Patched Image
      │
      ▼
YOLOv8 Detector
      │
      ▼
Detection Parser
      │
      ▼
Attack Target
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
Cosine LR Scheduler
      │
      ▼
Updated Patch
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
├── physical/
│
├── defense/
│
├── tests/
│
├── data/
│
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   ├── logs/
│   └── patches/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Module Responsibilities

## Configuration

Responsible for:

- Loading experiment configuration
- Device selection
- Hyperparameter management
- Experiment reproducibility
- Scheduler configuration

---

## Dataset

Responsible for:

- Loading COCO images
- Image preprocessing
- Tensor conversion
- Dataset abstraction
- DataLoader compatibility

---

## Adversarial Patch

Responsible for:

- Learnable patch representation
- Multiple initialization strategies
- Gradient optimization
- Pixel value clamping
- Checkpoint serialization

Supported initialization strategies:

- Gray
- Random
- Checkerboard
- Gaussian

---

## Expectation over Transformation (EOT)

Responsible for improving robustness during optimization.

Current transformations:

- Rotation
- Scaling

Framework prepared for future support of:

- Brightness
- Contrast
- Perspective
- Motion blur
- Noise

---

## Patch Applier

Responsible for:

- Overlaying the adversarial patch
- Random patch placement
- Boundary checking
- Batch compatibility
- Pixel clamping

---

## YOLO Detector

Responsible for:

- Loading YOLOv8
- Differentiable forward pass
- Configurable confidence threshold
- Configurable IoU threshold
- Device abstraction

---

## Detection Parser

Responsible for:

- Parsing detector outputs
- Producing standardized detection tensors
- Supporting evaluation and visualization

---

## Attack Target

Responsible for:

- Extracting optimization targets
- Selecting Top-K detections
- Isolating detector-specific logic
- Providing detector-independent interfaces for future attack objectives

---

## Loss Functions

Current implementation includes:

- Person Suppression Loss
- Confidence Loss
- Objectness Loss
- Total Variation Loss
- Non-Printability Score (placeholder)

The primary optimization objective minimizes person detection confidence while preserving differentiability.

---

## Trainer

Responsible for:

- Multi-epoch optimization
- Batch processing
- Random patch placement
- Forward propagation
- Loss computation
- Gradient computation
- Optimizer updates
- Learning rate scheduling
- Patch statistics
- Checkpoint management
- Loss history generation

---

## Evaluation

Responsible for:

- Original image inference
- Patched image inference
- Detection comparison
- Suppression metrics
- Confidence statistics
- Detection visualization
- Side-by-side comparison images
- CSV export
- JSON export

---

# Training Workflow

```text
Load Configuration
        │
        ▼
Load Dataset
        │
        ▼
Initialize Patch
        │
        ▼
Initialize Optimizer
        │
        ▼
Initialize Scheduler
        │
        ▼
Epoch Loop
        │
        ▼
Apply EOT
        │
        ▼
Random Patch Placement
        │
        ▼
Forward Through YOLOv8
        │
        ▼
Extract Targets
        │
        ▼
Compute Suppression Loss
        │
        ▼
Backpropagation
        │
        ▼
Optimizer Step
        │
        ▼
Scheduler Step
        │
        ▼
Clamp Patch
        │
        ▼
Save Metrics
        │
        ▼
Save Checkpoint
```

---

# Evaluation Workflow

```text
Original Image
      │
      ▼
YOLOv8 Inference
      │
      ▼
Original Metrics
      │
      ├───────────────┐
      ▼               │
Apply Patch           │
      ▼               │
Patched Image         │
      ▼               │
YOLOv8 Inference      │
      ▼               │
Patched Metrics       │
      └───────┬───────┘
              ▼
Comparison Metrics
              ▼
CSV Export
              ▼
JSON Export
              ▼
Detection Visualizations
```

---

# Design Principles

The framework is designed to be:

- **Modular** – each component has a single responsibility.
- **Extensible** – new detectors, losses, and transformations can be integrated easily.
- **Reproducible** – experiments are configuration-driven.
- **Research-Oriented** – suitable for benchmarking adversarial attacks.
- **Maintainable** – clear separation of training, evaluation, and attack logic.
- **Scalable** – prepared for larger datasets and additional attack objectives.

---

# Current Status (Commit 20)

## Completed

- Configuration System
- COCO Dataset Loader
- YOLOv8 Integration
- Detection Parser
- Adversarial Patch Module
- Multiple Patch Initialization Strategies
- Random Patch Placement
- Patch Application
- Attack Target Extraction
- Person Suppression Loss
- EOT (Rotation + Scaling)
- Multi-Epoch Training
- Adam Optimization
- Cosine Learning Rate Scheduler
- Checkpoint Saving
- Loss History Logging
- Evaluation Framework
- Detection Visualization
- CSV Export
- JSON Export

---

# Future Extensions

Planned improvements include:

- Best checkpoint selection
- Training loss visualization
- Multi-image optimization
- Multi-image evaluation
- Stronger EOT transformations
- Physical-world evaluation
- Attack Success Rate (ASR)
- Detector transferability experiments
- Benchmarking across YOLO variants