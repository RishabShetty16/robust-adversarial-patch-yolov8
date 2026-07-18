# Project Architecture

This document describes the overall software architecture of the **Robust Adversarial Patch Attack for YOLOv8** framework.

The project follows a modular, research-oriented design where each component is responsible for a single task. This allows individual modules to be replaced or extended without affecting the rest of the pipeline.

---

# Stage A Architecture

The current implementation focuses on building a fully differentiable optimization pipeline for adversarial patch training.

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
      
# High-Level Design

```
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
                    Patch Applier
                          │
                          ▼
                    Patched Image
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
```

---

# Repository Structure

```
attack/
│
├── configs/
│     └── default.yaml
│
├── attack_target.py
├── config.py
├── dataset.py
├── detector.py
├── losses.py
├── parser.py
├── patch.py
├── patch_applier.py
├── trainer.py
└── utils.py

experiments/
│
├── stage_a_baseline.py
└── stage_a_train.py

evaluation/

physical/

defense/

tests/

data/

outputs/
├── checkpoints/
├── figures/
├── logs/
└── patches/
```

---

## Training Infrastructure

The training engine is responsible for:

- Multi-epoch optimization
- Batch processing
- Gradient computation
- Optimizer updates
- Loss tracking
- Patch statistics
- Checkpoint management

# Module Responsibilities

## Configuration

Responsible for:

- Loading experiment configuration
- Device selection
- Parameter validation
- Experiment summary

---

## Dataset

Responsible for:

- Loading COCO images
- Image preprocessing
- Tensor conversion
- Dataset abstraction

---

## Patch

Responsible for:

- Learnable adversarial patch
- Initialization
- Clamping
- Saving
- Visualization

---

## Patch Applier

Responsible for:

- Overlaying the adversarial patch
- Boundary checking
- Pixel clamping
- Batch compatibility

---

## YOLO Detector

Responsible for:

- Loading YOLOv8
- Differentiable forward pass
- Standard inference interface

---

## Detection Parser

Responsible for:

- Parsing inference outputs
- Producing standardized detection dictionaries

**Note:** This module is primarily used during evaluation and baseline inference rather than the optimization pipeline.

---

## Attack Target

Responsible for:

- Extracting optimization targets from detector outputs
- Isolating detector-specific logic from the trainer
- Providing a stable interface for future attack objectives

---

## Loss Functions

Current losses include:

- Baseline optimization loss
- Confidence loss
- Objectness loss
- Total Variation (TV) loss
- Non-Printability Score (NPS) placeholder

Future versions will include detector-specific adversarial objectives.

---

## Trainer

Responsible for:

- Patch optimization
- Forward pass
- Loss computation
- Backpropagation
- Optimizer updates

Future commits will extend this module with:

- Multi-epoch training
- Batch processing
- Checkpointing
- Logging

---

# Design Principles

The repository is designed to be:

- **Modular** — each module has a single responsibility.
- **Reproducible** — configuration-driven experiments.
- **Extensible** — components can be replaced independently.
- **Research-oriented** — suitable for experimentation and benchmarking.
- **Maintainable** — clear separation between training, evaluation, and attack logic.

---

# Current Status

## ✅ Completed

- Configuration Management
- Dataset Loader
- Adversarial Patch Module
- Patch Applier
- YOLO Detector Wrapper
- Detection Parser
- Loss Function Framework
- Attack Target Abstraction
- Differentiable Optimization Pipeline
- Gradient Verification

---

## 🚧 Next Stage

The next development milestone introduces:

- Multi-epoch training loop
- DataLoader integration
- Logging
- Checkpoint saving
- Detector-specific person suppression loss
- Expectation over Transformation (EOT)