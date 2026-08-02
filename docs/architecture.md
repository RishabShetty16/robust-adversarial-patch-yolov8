# Software Architecture

This document describes the software architecture of the **Robust Adversarial Patch Attack Framework for YOLOv8**.

The framework follows a **modular, research-oriented architecture**, where each module is responsible for a single well-defined task within the adversarial patch optimization pipeline.

The primary design goals are:

- Modularity
- Reproducibility
- Extensibility
- Maintainability
- End-to-end differentiability
- Research experimentation

Unlike monolithic implementations, every stage of the attack pipeline has been isolated into reusable components, allowing researchers to independently modify optimization objectives, placement strategies, evaluation metrics, and transformation pipelines.

---

# Project Overview

The objective of this framework is to learn a **universal adversarial patch** capable of suppressing **person detections** produced by the YOLOv8 object detector.

The optimization process is fully differentiable, allowing gradients to propagate directly from detector predictions back to the adversarial patch.

The framework currently supports

- Universal adversarial patch optimization
- YOLOv8 integration
- COCO dataset training
- Adaptive patch placement
- Person-aware placement
- Composite configurable loss functions
- Confidence-weighted suppression
- Expectation over Transformation (EOT)
- Automatic checkpoint management
- Comprehensive evaluation
- Research-oriented experimentation

The architecture has been designed to support future extensions including physical-world attacks, detector transferability, additional datasets, and stronger robustness transformations.

---

# Architectural Goals

The framework has been developed around the following software engineering principles.

## Modularity

Each module performs a single task.

Examples include

- Dataset loading
- Patch generation
- Detector inference
- Loss computation
- Evaluation

This separation reduces coupling between components and simplifies future development.

---

## Extensibility

New functionality can be added without modifying the existing training pipeline.

Examples include

- New detectors
- New datasets
- New optimization objectives
- New placement strategies
- Additional transformations
- Alternative evaluation metrics

---

## Reproducibility

All experiments are configuration-driven.

Every experiment can be reproduced simply by sharing the configuration file without modifying the implementation.

The configuration system controls

- Dataset
- Optimizer
- Scheduler
- Patch
- Placement
- Losses
- EOT
- Evaluation

---

## Research-Oriented Design

Unlike production inference systems, this framework is intended for adversarial machine learning research.

The implementation prioritizes

- Flexibility
- Experimentation
- Explainability
- Repeatability

This enables rapid benchmarking of different adversarial optimization strategies.

---

## Maintainability

Each module has clearly defined responsibilities.

The architecture minimizes dependencies between unrelated components, making debugging and future improvements significantly easier.

---

# High-Level Architecture

The complete adversarial patch optimization framework consists of five major subsystems.

```text
                   Configuration System
                           │
                           ▼
                 Training Pipeline
                           │
                           ▼
                Optimization Engine
                           │
                           ▼
                Evaluation Framework
                           │
                           ▼
                 Experiment Analysis
```

Each subsystem communicates through clearly defined interfaces while remaining independently maintainable.

---

# Overall System Architecture

The overall architecture of the framework is illustrated below.

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
                    Universal Adversarial Patch
                                     │
                                     ▼
              Expectation over Transformation (EOT)
                                     │
                                     ▼
                    Adaptive Patch Placement Engine
                                     │
                     ┌───────────────┼────────────────┐
                     │               │                │
                     ▼               ▼                ▼
                  Fixed           Random         Person-aware
                                                      │
                                                      ▼
                                    Head • Torso • Center • Feet
                                                      │
                                                      ▼
                      Differentiable Patch Application
                                     │
                                     ▼
                           Patched Image Tensor
                                     │
                                     ▼
                             YOLOv8 Detector
                                     │
                                     ▼
                          Detection Parsing Module
                                     │
                                     ▼
                         Attack Target Selection
                                     │
                                     ▼
                  Confidence-weighted Suppression
                                     │
                                     ▼
                        Composite Optimization Loss
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
                           Updated Patch Parameters
                                     │
                ┌────────────────────┴────────────────────┐
                ▼                                         ▼
       Checkpoint Management                    Training Statistics
                │                                         │
                ▼                                         ▼
          Best Model Saved                  Loss History & Curves
```

---

# Data Flow Overview

During training, information flows sequentially through the framework.

```text
Configuration
      │
      ▼
Dataset
      │
      ▼
Input Images
      │
      ▼
Universal Patch
      │
      ▼
Expectation over Transformation
      │
      ▼
Adaptive Placement
      │
      ▼
Patch Application
      │
      ▼
YOLOv8 Forward Pass
      │
      ▼
Detection Parsing
      │
      ▼
Attack Target Selection
      │
      ▼
Composite Loss
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

Each stage consumes the output of the previous stage while preserving differentiability throughout the optimization process.

---

# Core Components

The framework consists of five primary architectural layers.

```text
Layer 1

Configuration

↓

Layer 2

Data Processing

↓

Layer 3

Attack Generation

↓

Layer 4

Optimization

↓

Layer 5

Evaluation
```

Each layer contains multiple independent modules responsible for specific tasks.

This layered architecture simplifies future expansion and allows individual components to evolve independently without affecting the rest of the framework.

---

# Repository Organization

The project is organized into independent modules, each responsible for a specific stage of the adversarial patch optimization pipeline.

```text
robust-adversarial-patch-yolov8/

│
├── attack/
│   ├── configs/
│   │   └── default.yaml
│   │
│   ├── attack_target.py
│   ├── bbox_utils.py
│   ├── config.py
│   ├── dataset.py
│   ├── detector.py
│   ├── eot.py
│   ├── losses.py
│   ├── parser.py
│   ├── patch.py
│   ├── patch_applier.py
│   ├── placement.py
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
│   ├── evaluation/
│   ├── figures/
│   ├── logs/
│   └── patches/
│
├── data/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

The repository follows a layered architecture where each directory represents a major subsystem of the framework.

---

# Architecture Layers

The framework is divided into five logical layers.

```text
Configuration Layer

↓

Data Processing Layer

↓

Attack Generation Layer

↓

Optimization Layer

↓

Evaluation Layer
```

Each layer communicates only with the layers immediately above and below it, reducing coupling between unrelated components.

---

# Configuration Layer

## Purpose

The configuration layer centralizes all experiment parameters.

Rather than hardcoding values inside the implementation, every important hyperparameter is stored inside

```text
attack/configs/default.yaml
```

This design enables reproducible experiments and simplifies hyperparameter tuning.

---

## Configuration Responsibilities

The configuration system manages

- Experiment name
- Random seed
- Device selection
- Dataset parameters
- Model parameters
- Patch parameters
- Optimizer settings
- Scheduler settings
- Placement strategy
- EOT transformations
- Composite loss
- Suppression parameters
- Evaluation settings

---

## Advantages

Using a configuration-driven architecture provides

- Reproducibility
- Cleaner code
- Easier experimentation
- Separation between implementation and hyperparameters

---

# Dataset Layer

**Primary Module**

```text
attack/dataset.py
```

## Responsibilities

The dataset module prepares all images used during optimization.

Its responsibilities include

- Reading COCO images
- Image preprocessing
- Tensor conversion
- Batch preparation
- PyTorch Dataset implementation
- DataLoader compatibility

---

## Input

```text
COCO Images
```

---

## Output

```text
Tensor Batch

Shape

(B, C, H, W)
```

---

## Design Decisions

The dataset implementation follows the standard PyTorch Dataset abstraction.

Benefits include

- Easy batching
- Multi-worker loading
- GPU compatibility
- Scalability

---

# Detector Layer

**Primary Module**

```text
attack/detector.py
```

The detector module serves as an abstraction layer between the optimization framework and YOLOv8.

The remaining architecture never communicates directly with Ultralytics.

Instead, every interaction occurs through this wrapper.

---

## Responsibilities

- Load YOLOv8 weights
- Device management
- Forward inference
- Prediction interface
- Differentiable forward pass
- Confidence filtering
- IoU thresholding

---

## Input

```text
Image Tensor
```

---

## Output

```text
YOLO Predictions
```

---

## Benefits

The abstraction layer allows future replacement of YOLOv8 with alternative detectors without modifying the training pipeline.

Potential future detectors include

- YOLOv9
- YOLOv10
- RT-DETR
- Faster R-CNN

---

# Universal Patch Module

**Primary Module**

```text
attack/patch.py
```

The patch module represents the learnable adversarial patch.

Unlike ordinary tensors, the patch is implemented as a trainable neural network parameter.

---

## Responsibilities

- Patch initialization
- Parameter optimization
- Pixel clamping
- Checkpoint serialization

---

## Initialization Strategies

Currently implemented

- Gray
- Random
- Gaussian
- Checkerboard

Each initialization strategy provides a different optimization starting point.

---

## Output

```text
Patch Tensor

Shape

(3, Patch Size, Patch Size)
```

---

# Patch Placement Module

**Primary Module**

```text
attack/placement.py
```

Patch placement determines where the adversarial patch is inserted into each image.

Rather than relying on fixed coordinates, the framework supports several placement strategies.

---

## Supported Modes

```text
Fixed

↓

Random

↓

Person-aware
```

---

## Fixed Placement

Uses predefined image coordinates.

Suitable for

- Debugging
- Controlled experiments
- Reproducibility

---

## Random Placement

Randomly samples valid patch coordinates.

Advantages

- Increased robustness
- Better generalization
- Reduced overfitting

---

## Person-aware Placement

Person-aware placement first identifies the largest detected person and then positions the patch relative to configurable body anchors.

Supported anchors

- Center
- Head
- Torso
- Feet

If no person is detected, the placement engine automatically falls back to center placement.

---

# Bounding Box Utilities

**Primary Module**

```text
attack/bbox_utils.py
```

This module provides reusable utilities for processing detector outputs.

---

## Responsibilities

- Extract person detections
- Compute bounding-box dimensions
- Find largest detected person
- Compute anchor coordinates

---

## Why Separate This Module?

Bounding-box processing is used by multiple components.

Keeping these utilities independent improves

- Reusability
- Readability
- Maintainability

and prevents duplicated logic across the project.

---

# Patch Application Module

**Primary Module**

```text
attack/patch_applier.py
```

The patch applier inserts the adversarial patch into the input image tensor.

Unlike image editing libraries, this implementation is fully differentiable.

---

## Responsibilities

- Patch overlay
- Boundary checking
- Batch compatibility
- Gradient preservation

---

## Input

```text
Image Tensor

+

Patch Tensor

+

Placement Coordinates
```

---

## Output

```text
Patched Image Tensor
```

Because gradients are preserved during the overlay operation, optimization can update the patch parameters through backpropagation.

---

# Optimization Layer

The optimization layer is responsible for learning the adversarial patch through end-to-end differentiable optimization.

Unlike traditional machine learning pipelines where model parameters are optimized, this framework **keeps the YOLOv8 detector frozen** and optimizes **only the adversarial patch**.

The optimization process combines several independent modules into a unified training pipeline.

```text
Universal Patch
        │
        ▼
Expectation over Transformation
        │
        ▼
Adaptive Patch Placement
        │
        ▼
Patch Application
        │
        ▼
YOLOv8 Forward Pass
        │
        ▼
Detection Parsing
        │
        ▼
Attack Target Selection
        │
        ▼
Composite Loss
        │
        ▼
Backpropagation
        │
        ▼
Patch Update
```

---

# Expectation over Transformation (EOT)

**Primary Module**

```text
attack/eot.py
```

Expectation over Transformation (EOT) improves the robustness of the adversarial patch by applying random transformations during training.

Instead of optimizing the patch for a single appearance, the framework optimizes over a distribution of transformed patches.

---

## Purpose

The objective of EOT is to produce patches that remain effective under different physical and geometric conditions.

Examples include

- Different viewing angles
- Different distances
- Small rotations
- Scale changes

This improves the likelihood that the patch remains effective outside the exact training conditions.

---

## Current Transformations

The framework currently implements

### Rotation

Randomly rotates the patch within a configurable range.

```text
Patch

↓

Random Rotation

↓

Rotated Patch
```

---

### Scaling

Randomly enlarges or shrinks the patch before applying it.

```text
Patch

↓

Random Scaling

↓

Scaled Patch
```

Scaling improves robustness to viewing distance.

---

## Planned Transformations

The EOT module has been designed to support future additions without changing the rest of the framework.

Examples include

- Brightness
- Contrast
- Gaussian Noise
- Motion Blur
- Perspective Transformation
- Color Jitter
- Camera Distortion

---

# Attack Target Selection

**Primary Module**

```text
attack/attack_target.py
```

Rather than optimizing every detector prediction, the framework selects only detections relevant to the attack objective.

Currently, the framework targets the **person** class.

---

## Responsibilities

- Filter person detections
- Extract confidence scores
- Top-K selection
- Batch-wise processing

---

## Workflow

```text
YOLO Predictions
        │
        ▼
Filter Person Class
        │
        ▼
Confidence Extraction
        │
        ▼
Top-K Selection
        │
        ▼
Optimization Targets
```

Only these selected detections contribute to the optimization loss.

---

## Advantages

This approach

- Reduces optimization noise
- Focuses gradients on relevant detections
- Improves convergence
- Simplifies future attack objectives

---

# Composite Loss Framework

**Primary Module**

```text
attack/losses.py
```

The framework implements a modular composite loss function.

Instead of relying on a single objective, multiple loss components are combined into one optimization objective.

---

## Overall Objective

```text
Total Loss

=

Suppression Loss

+

TV Weight × Total Variation Loss

+

NPS Weight × Non-Printability Score
```

Each component is independently configurable through the experiment configuration.

---

# Person Suppression Loss

The primary optimization objective minimizes detector confidence for person detections.

Current implementation

```text
Loss

=

mean(confidence^power)
```

where

- confidence = detector confidence score
- power = configurable exponent

---

## Confidence Weighting

The exponent controls how strongly high-confidence detections influence optimization.

Examples

```text
Power = 1

Linear weighting
```

```text
Power = 2

Quadratic weighting
```

```text
Power = 3

Cubic weighting
```

Increasing the exponent causes the optimization process to focus more strongly on difficult, high-confidence detections.

---

## Configurable Parameters

Current parameters

```yaml
loss:

  suppression:

    threshold: 0.25

    power: 2.0
```

This allows suppression behavior to be modified without changing the implementation.

---

# Total Variation Loss

Total Variation (TV) encourages spatial smoothness in the adversarial patch.

Large neighboring pixel differences are penalized, reducing high-frequency noise.

Conceptually

```text
Smooth Patch

↓

Low TV Loss
```

```text
Noisy Patch

↓

High TV Loss
```

The influence of TV regularization is controlled through the configuration file.

---

# Non-Printability Score (NPS)

The Non-Printability Score encourages the patch to contain colors that are easier to reproduce with physical printers.

Current implementation provides the framework for future physical-world optimization.

The contribution of NPS is controlled independently through

```yaml
loss:

  nps_weight
```

---

# Composite Loss Configuration

The framework exposes all optimization weights through the configuration file.

Example

```yaml
loss:

  suppression_weight: 1.0

  tv_weight: 0.0

  nps_weight: 0.001

  suppression:

    threshold: 0.25

    power: 2.0
```

This enables rapid experimentation without modifying source code.

---

# Training Engine

**Primary Module**

```text
attack/trainer.py
```

The trainer orchestrates the complete optimization process.

It coordinates every component involved in adversarial patch learning.

---

## Responsibilities

- Load configuration
- Initialize optimizer
- Initialize scheduler
- Multi-epoch optimization
- Batch processing
- Patch placement
- Detector inference
- Loss computation
- Gradient computation
- Optimizer update
- Patch clamping
- Logging
- Checkpoint management

---

# Optimizer

Current optimizer

```text
Adam
```

Responsibilities

- Gradient updates
- Stable optimization
- Adaptive learning rates

---

# Learning Rate Scheduler

Current scheduler

```text
Cosine Annealing
```

Responsibilities

- Smooth learning-rate decay
- Stable convergence
- Reduced oscillation

---

# Gradient Flow

The framework preserves gradients across the entire optimization pipeline.

```text
Patch
      │
      ▼
Patch Application
      │
      ▼
YOLO Forward Pass
      │
      ▼
Target Selection
      │
      ▼
Composite Loss
      │
      ▼
Backpropagation
      │
      ▼
Patch Parameters
```

Only the adversarial patch receives parameter updates.

YOLOv8 remains frozen throughout optimization.

---

# Optimization Workflow

Each optimization step follows the sequence below.

```text
Load Mini-batch
        │
        ▼
Transform Patch (EOT)
        │
        ▼
Compute Placement
        │
        ▼
Apply Patch
        │
        ▼
YOLOv8 Forward Pass
        │
        ▼
Parse Detections
        │
        ▼
Select Targets
        │
        ▼
Compute Composite Loss
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
Save Statistics
```

---

# Training Outputs

During optimization, the framework automatically records

- Current epoch
- Learning rate
- Patch statistics
- Suppression loss
- TV loss
- NPS loss
- Total loss
- Gradient information
- Best checkpoint
- Loss history

Generated outputs include

```text
outputs/

├── checkpoints/
│   ├── best.pt
│   └── latest.pt
│
├── logs/
│   └── loss_history.csv
│
├── figures/
│   └── training_loss.png
```

These artifacts provide complete visibility into the optimization process and facilitate reproducible experimentation.

---

# Evaluation Layer

The evaluation layer measures the effectiveness of the optimized adversarial patch on previously unseen images.

Unlike the training pipeline, which focuses on optimization, the evaluation framework focuses on **quantifying attack performance** using multiple metrics and visual analysis.

The evaluation framework is completely independent of the training engine, allowing trained patches to be evaluated repeatedly under different configurations.

---

# Evaluation Architecture

The evaluation pipeline follows the architecture shown below.

```text
                     Evaluation Dataset
                            │
                            ▼
                     Original Image
                            │
                            ▼
                     YOLOv8 Inference
                            │
                            ▼
                  Original Detection Results
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          ▼                                   ▼
Apply Adversarial Patch                Save Original Results
          │
          ▼
                   Patched Image
                            │
                            ▼
                     YOLOv8 Inference
                            │
                            ▼
                  Patched Detection Results
                            │
                            ▼
                 Detection Comparison
                            │
                            ▼
                 Metric Computation
                            │
                            ▼
             Visualization Generation
                            │
                            ▼
            CSV / JSON Result Export
```

The framework evaluates both the original and patched image using identical detector settings to ensure a fair comparison.

---

# Evaluation Module

**Primary Module**

```text
evaluation/evaluate_patch.py
```

The evaluation module orchestrates the complete benchmarking pipeline.

---

## Responsibilities

- Load trained patch
- Load evaluation dataset
- Perform original inference
- Apply adversarial patch
- Perform patched inference
- Compute metrics
- Generate visualizations
- Export experiment results

---

# Evaluation Workflow

Each image passes through the following stages.

```text
Image
   │
   ▼
Original Detection
   │
   ▼
Patch Placement
   │
   ▼
Patch Application
   │
   ▼
Patched Detection
   │
   ▼
Metric Computation
   │
   ▼
Visualization
   │
   ▼
Export
```

This process is repeated for every image in the evaluation dataset.

---

# Metric Computation

**Primary Module**

```text
evaluation/metrics.py
```

The metrics module computes quantitative measurements describing the effectiveness of the adversarial attack.

All metrics are computed for individual images and aggregated across the entire dataset.

---

## Average Suppression

Measures the reduction in detected persons after applying the adversarial patch.

```text
Suppression

=

Original Persons − Patched Persons
----------------------------------
Original Persons
```

Higher suppression indicates a stronger attack.

---

## Confidence Drop

Measures the reduction in detector confidence after patch application.

```text
Confidence Drop

=

Original Confidence − Patched Confidence
```

Higher values indicate that the detector becomes less confident.

---

## Detection Retention

Measures how many original detections remain after the attack.

```text
Retention

=

Patched Persons
---------------
Original Persons
```

Lower retention indicates a stronger adversarial attack.

---

## Attack Success Rate

The Attack Success Rate (ASR) measures the proportion of evaluated images for which the adversarial patch successfully achieves the attack objective.

Higher ASR corresponds to a more successful attack.

---

# Dataset-Level Statistics

Rather than evaluating only a single image, the framework computes aggregate statistics across the evaluation dataset.

Example summary

```text
============================================================

Dataset Evaluation Summary

============================================================

Images Evaluated : 50

Average Suppression : 15.98%

Average Confidence Drop : 17.27%

Average Retention : 38.02%

Attack Success Rate : 26.00%

============================================================
```

These statistics provide a comprehensive overview of attack performance.

---

# Visualization Pipeline

**Primary Module**

```text
evaluation/visualization.py
```

Visualization is an important component of adversarial attack analysis.

The framework automatically generates annotated images showing detector predictions before and after applying the adversarial patch.

---

## Generated Visualizations

The evaluation pipeline automatically produces

- Original detection image
- Patched detection image
- Side-by-side comparison

Example

```text
outputs/

└── evaluation/

    ├── original_detection.jpg

    ├── patched_detection.jpg

    └── comparison.jpg
```

These visualizations complement the quantitative evaluation metrics.

---

# Export Pipeline

**Primary Module**

```text
evaluation/export.py
```

The export module stores evaluation statistics for later analysis.

Current supported formats

- CSV
- JSON

Example

```text
outputs/

└── evaluation/

    ├── results.csv

    └── results.json
```

These files can be used for

- plotting
- experiment comparison
- report generation
- statistical analysis

---

# Experiment Framework

The architecture is designed to support reproducible research.

Experiments are driven entirely by the configuration system.

Researchers can modify

- placement strategy
- suppression parameters
- optimizer
- scheduler
- EOT
- composite loss

without modifying the implementation.

---

# Placement Experiments

The framework supports multiple patch placement strategies.

```text
Fixed

↓

Random

↓

Person-aware

        │

        ▼

Head

Torso

Center

Feet
```

This enables systematic comparison of different patch locations.

---

# Composite Loss Experiments

The optimization objective supports multiple configurable components.

```text
Suppression Loss

+

TV Loss

+

Non-Printability Score
```

Each component can be independently enabled, disabled, or re-weighted.

This enables detailed ablation studies.

---

# Suppression Power Experiments

The confidence-weighted suppression objective introduces an adjustable exponent.

```text
Loss

=

mean(confidence^power)
```

Current evaluated values include

```text
Power = 1

Power = 2

Power = 3
```

These experiments determine how strongly high-confidence detections should influence optimization.

---

# Current Experimental Pipeline

The current research workflow is summarized below.

```text
Configuration
       │
       ▼
Training
       │
       ▼
Checkpoint
       │
       ▼
Evaluation
       │
       ▼
Metric Computation
       │
       ▼
Visualization
       │
       ▼
CSV / JSON Export
       │
       ▼
Experiment Analysis
```

---

# Data Logging

The framework automatically records experiment information throughout training and evaluation.

Training logs include

- Epoch
- Learning rate
- Suppression loss
- TV loss
- NPS loss
- Total loss
- Patch statistics

Evaluation logs include

- Average suppression
- Confidence drop
- Retention
- Attack Success Rate

This information supports reproducibility and enables comparison across multiple experiments.

---

# Evaluation Outputs

The complete evaluation pipeline generates the following outputs.

```text
outputs/

├── checkpoints/
│
├── evaluation/
│   ├── original_detection.jpg
│   ├── patched_detection.jpg
│   ├── comparison.jpg
│   ├── results.csv
│   └── results.json
│
├── figures/
│   └── training_loss.png
│
├── logs/
│   └── loss_history.csv
│
└── patches/
```

The separation of checkpoints, logs, figures, and evaluation artifacts simplifies experiment management and allows multiple training runs to be compared efficiently.

---

# Design Principles

The framework has been designed according to modern software engineering practices while addressing the unique requirements of adversarial machine learning research.

The following principles guided the architecture.

---

## Modular Design

Every component is responsible for exactly one major task.

Examples include

- Dataset loading
- Patch optimization
- Patch placement
- Loss computation
- Detector abstraction
- Evaluation
- Visualization

This minimizes coupling between modules and makes the framework significantly easier to extend.

---

## Separation of Concerns

Training, evaluation, optimization, visualization, and experiment management are implemented independently.

```text
Training

↓

Evaluation

↓

Visualization

↓

Experiment Analysis
```

Changes to one subsystem do not require modifications to unrelated components.

---

## Configuration-Driven Experiments

The framework separates implementation from experimentation.

All experiment parameters are defined inside

```text
attack/configs/default.yaml
```

This allows researchers to reproduce experiments by sharing configuration files rather than modifying source code.

Current configurable components include

- Dataset
- Detector
- Optimizer
- Scheduler
- Patch
- Placement
- EOT
- Composite Loss
- Suppression Parameters
- Evaluation

---

## Detector Abstraction

The optimization pipeline never communicates directly with Ultralytics.

Instead,

```text
Trainer

↓

Detector Wrapper

↓

YOLOv8
```

This abstraction makes it possible to integrate future object detectors with minimal changes to the training pipeline.

Potential future detectors include

- YOLOv9
- YOLOv10
- RT-DETR
- DETR
- Faster R-CNN

---

## Reusability

Reusable functionality is isolated into dedicated modules.

Examples include

```text
bbox_utils.py

placement.py

metrics.py

export.py

visualization.py
```

This avoids duplicated logic and improves maintainability.

---

## Scalability

The architecture has been designed with future expansion in mind.

The current implementation can be extended to support

- Additional datasets
- Larger image resolutions
- Multi-GPU training
- Distributed optimization
- Physical-world attacks
- Detector transferability
- Video-based evaluation

without requiring major architectural changes.

---

# Software Engineering Decisions

Several architectural decisions were made to simplify future development.

---

## Why a Universal Patch?

Optimizing a single universal patch

- reduces memory usage
- simplifies deployment
- supports physical-world attacks
- improves reproducibility

---

## Why Composite Loss?

Different adversarial objectives often compete.

A configurable composite loss allows

- rapid experimentation
- ablation studies
- future loss integration

without modifying the trainer.

---

## Why Adaptive Placement?

Real-world adversarial patches cannot always be placed at fixed coordinates.

Person-aware placement enables the optimization process to focus on semantically meaningful body regions such as

- head
- torso
- center
- feet

making the framework more suitable for practical attack scenarios.

---

## Why Confidence-weighted Suppression?

Not all detections are equally important.

High-confidence detections typically represent stronger predictions.

Confidence-weighted suppression encourages the optimizer to prioritize these detections, resulting in more effective adversarial patches.

---

## Why End-to-End Differentiability?

The framework preserves gradients throughout the optimization pipeline.

```text
Patch

↓

Patch Application

↓

YOLOv8

↓

Target Selection

↓

Composite Loss

↓

Backpropagation
```

This enables efficient gradient-based optimization without requiring custom approximation techniques.

---

# Current Architecture Status

## Framework Status

**Research Framework — Commit 31**

The framework currently implements a complete adversarial patch optimization pipeline.

Implemented components include

- Configuration system
- COCO dataset loader
- YOLOv8 detector wrapper
- Universal adversarial patch
- Multiple patch initialization strategies
- Adaptive patch placement
- Person-aware placement
- Bounding box utilities
- Differentiable patch application
- Expectation over Transformation
- Attack target extraction
- Confidence-weighted suppression
- Composite configurable loss
- Adam optimization
- Cosine learning rate scheduling
- Automatic checkpoint saving
- Best checkpoint selection
- Loss history logging
- Training curve visualization
- Dataset-wide evaluation
- Visualization pipeline
- CSV export
- JSON export

---

# Completed Architecture

The framework currently consists of

```text
Configuration Layer

↓

Dataset Layer

↓

Patch Generation Layer

↓

Transformation Layer

↓

Placement Layer

↓

Detector Layer

↓

Target Selection Layer

↓

Optimization Layer

↓

Evaluation Layer

↓

Visualization Layer

↓

Experiment Analysis Layer
```

This layered architecture provides clear boundaries between independent subsystems while supporting future expansion.

---

# Future Architecture Roadmap

The architecture has been designed to accommodate future research directions.

---

## Transformation Layer

Planned additions include

- Brightness
- Contrast
- Perspective warp
- Gaussian noise
- Motion blur
- Weather simulation
- Camera distortion

---

## Optimization Layer

Future improvements include

- Adaptive loss weighting
- Dynamic target selection
- Multi-objective optimization
- Mixed precision training
- Distributed optimization

---

## Evaluation Layer

Future evaluation capabilities include

- Physical-world benchmarking
- Video-based evaluation
- Cross-dataset benchmarking
- Cross-detector transferability
- Patch robustness benchmarking

---

## Detector Layer

Planned detector support includes

- YOLOv9
- YOLOv10
- RT-DETR
- DETR
- Faster R-CNN

---

## Experiment Framework

Future experiments will investigate

- Larger patch sizes
- Multi-scale optimization
- Stronger EOT strategies
- Patch transferability
- Physical printability
- Robustness under environmental variation

---

# Architecture Summary

The **Robust Adversarial Patch Attack Framework for YOLOv8** has been developed as a modular, extensible, and research-oriented software system.

The architecture emphasizes

- modularity
- reproducibility
- configurability
- maintainability
- end-to-end differentiability

Every subsystem has a clearly defined responsibility, enabling independent development while maintaining seamless integration across the optimization pipeline.

By separating configuration, data processing, optimization, evaluation, and experiment management into dedicated modules, the framework provides a strong foundation for future research in adversarial machine learning, robust computer vision, and AI security.

The current implementation represents a complete end-to-end research framework capable of training, evaluating, and analyzing universal adversarial patches against YOLOv8 while remaining flexible enough to support future extensions and emerging research directions.

---

# Document Information

**Project:** Robust Adversarial Patch Attack Framework for YOLOv8

**Architecture Version:** Commit 31

**Programming Language:** Python

**Deep Learning Framework:** PyTorch

**Object Detector:** Ultralytics YOLOv8

**Primary Dataset:** COCO

**Optimization:** Adam + Cosine Annealing Learning Rate Scheduler

**Author:** Rishab Shetty

**Institution:** PES University

**Research Areas:**

- Adversarial Machine Learning
- Computer Vision
- Artificial Intelligence
- AI Security
- Deep Learning

---