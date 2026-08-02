# Development Log

This document records the evolution of the **Robust Adversarial Patch Attack Framework for YOLOv8** from its initial repository setup into a modular, research-oriented adversarial machine learning framework.

Rather than simply documenting code changes, this log captures the engineering decisions, architectural milestones, implementation details, and validation performed throughout the development process.

The project follows an incremental development methodology where each commit introduces a self-contained feature that is fully tested before becoming the foundation for subsequent work.

---

# Development Philosophy

The framework has been developed according to the following principles.

- Build small, testable modules.
- Verify every feature before introducing new functionality.
- Maintain a modular architecture.
- Preserve reproducibility through configuration-driven experiments.
- Keep the optimization pipeline fully differentiable.
- Produce research-quality software rather than a single proof-of-concept implementation.

Every major milestone includes:

- Objective
- Implementation
- Validation
- Outcome

---

# Commit 1 — Initialize Project Structure

## Commit Message

```text
Initialize project structure
```

---

## Objective

Establish a clean and modular repository structure before implementing any functionality.

A well-defined project layout simplifies future development, improves maintainability, and separates independent components into dedicated modules.

---

## Implemented

Created the initial Git repository.

Initialized version control.

Designed the repository hierarchy.

Created the following directories:

```text
attack/
defense/
evaluation/
physical/
data/
outputs/
notebooks/
tests/
assets/
docs/
```

Created output directories:

```text
outputs/

├── checkpoints/
├── figures/
├── logs/
└── patches/
```

Added the initial project files.

```text
README.md

requirements.txt

train.py

evaluate.py

.gitignore
```

Created the initial attack module.

```text
attack/

├── __init__.py
├── config.py
├── utils.py
├── dataset.py
├── detector.py
├── patch.py
└── configs/default.yaml
```

---

## Validation

Verified

- Repository structure
- Git initialization
- Directory organization
- Import compatibility

---

## Outcome

A clean project foundation was established, enabling independent development of future modules.

---

# Commit 2 — Configuration Management System

## Commit Message

```text
Implement configuration management module
```

---

## Objective

Remove hardcoded experiment parameters by introducing a centralized configuration system.

The goal was to make every experiment reproducible through a YAML configuration file.

---

## Implemented

Developed

```text
attack/config.py
```

Implemented functionality for

- YAML configuration loading
- Configuration validation
- Automatic device selection
- Experiment summary printing

Implemented the following functions

```python
load_config()

validate_config()

get_device()

print_config_summary()
```

Configuration parameters now include

- Experiment name
- Dataset settings
- Model settings
- Patch configuration
- Optimizer parameters
- Training settings
- Evaluation settings

---

## Validation

Successfully verified

- YAML parsing
- Configuration loading
- Device detection
- Configuration validation
- Summary generation

Example output

```text
============================================================
Experiment Configuration
============================================================

Experiment : baseline_patch_attack

Device     : auto

Model      : yolov8n.pt

Patch Size : 160

Optimizer  : Adam

LR         : 0.03

============================================================
```

---

## Outcome

All experiment parameters became configuration-driven, significantly improving reproducibility and simplifying future experimentation.

---

# Commit 3 — Utility Module

## Commit Message

```text
Implement utility module
```

---

## Objective

Develop reusable helper functions required throughout the project.

Rather than duplicating common functionality across modules, all shared utilities were centralized.

---

## Implemented

Created

```text
attack/utils.py
```

Implemented

- Random seed initialization
- Output directory creation
- Timestamp generation
- Model parameter counting

---

## Validation

Successfully verified

- Deterministic random seeds
- Automatic directory creation
- Timestamp formatting
- Parameter counting

---

## Outcome

The utility module became the common foundation used by the training and evaluation pipelines.

---

# Commit 4 — Universal Adversarial Patch Module

## Commit Message

```text
Implement universal adversarial patch module
```

---

## Objective

Implement the learnable adversarial patch that serves as the primary optimization variable throughout the framework.

Unlike detector parameters, the adversarial patch would become the only trainable component during optimization.

---

## Implemented

Created

```text
attack/patch.py
```

Implemented

- Learnable patch representation
- Gray initialization
- Random initialization
- Checkerboard initialization
- Pixel clamping
- Patch statistics
- Patch serialization
- Patch loading
- Patch visualization

The patch was implemented as a PyTorch module, allowing gradients to propagate directly through the optimization pipeline.

---

## Validation

Verified

- Patch creation
- Initialization methods
- Pixel statistics
- Save functionality
- Load functionality
- Visualization generation

Example statistics

```text
Patch Shape

(3, 160, 160)

Mean

0.5000

Minimum

0.0000

Maximum

1.0000
```

---

## Outcome

A reusable universal adversarial patch module was successfully integrated into the framework.

---

# Commit 5 — COCO Dataset Module

## Commit Message

```text
Implement COCO dataset loader
```

---

## Objective

Develop a reusable dataset loader compatible with PyTorch for adversarial patch optimization.

The loader needed to support configurable image preprocessing while remaining independent of the optimization pipeline.

---

## Implemented

Created

```text
attack/dataset.py
```

Implemented

- COCO image discovery
- Image loading
- Resize transformation
- Tensor conversion
- Dataset abstraction
- Configuration-driven dataset paths

Added

```text
test_dataset.py
```

for independent validation.

---

## Validation

Successfully verified

- Dataset initialization
- Image loading
- Tensor conversion
- Batch compatibility
- Output tensor shapes

Example output

```text
============================================================
COCO Dataset Initialized
============================================================

Image Directory

data/coco/images

Image Size

640

Images Found

100

============================================================
```

---

## Outcome

The framework gained a reusable PyTorch-compatible dataset pipeline capable of supplying batches of images for future adversarial optimization experiments.

---

# Progress Summary

| Commit | Milestone | Status |
|---------|-----------|--------|
| 1 | Repository Structure | ✅ Complete |
| 2 | Configuration Management | ✅ Complete |
| 3 | Utility Module | ✅ Complete |
| 4 | Universal Patch Module | ✅ Complete |
| 5 | COCO Dataset Loader | ✅ Complete |

At this stage, the project had established a solid software foundation consisting of configuration management, reusable utilities, a learnable adversarial patch representation, and a configurable dataset pipeline. These components formed the basis for integrating differentiable patch application and detector-aware optimization in subsequent development phases.

# Commit 6 — Differentiable Patch Application Module

## Commit Message

```text
Implement differentiable patch application module
```

---

## Objective

Develop a reusable module capable of applying the adversarial patch to images while preserving gradient flow.

Since the adversarial patch is optimized through backpropagation, the overlay operation itself must remain fully differentiable.

---

## Implemented

Created

```text
attack/patch_applier.py
```

Implemented

- Single-image patch application
- Batch patch application
- Boundary-safe placement
- Pixel value clamping
- Differentiable tensor operations
- Gradient preservation

The module supports arbitrary patch positions while ensuring that the patch always remains inside image boundaries.

---

## Validation

Successfully verified

- Patch overlay
- Batch processing
- Tensor dimensions
- Pixel value ranges
- Gradient compatibility

Example output

```text
============================================================
Patch Application
============================================================

Input Shape

(3, 640, 640)

Patch Shape

(3, 160, 160)

Output Shape

(3, 640, 640)

Pixel Range

[0, 1]

============================================================
```

---

## Outcome

A reusable differentiable patch application module was successfully integrated into the framework.

---

# Commit 7 — YOLOv8 Detector Wrapper and Detection Parser

## Commit Message

```text
Implement YOLOv8 detector abstraction
```

---

## Objective

Create a detector abstraction layer that isolates the remainder of the framework from Ultralytics-specific implementation details.

This design simplifies future integration of additional object detectors.

---

## Implemented

Created

```text
attack/detector.py

attack/parser.py
```

Implemented

- Automatic YOLOv8 model loading
- Device abstraction
- Configurable confidence threshold
- Configurable IoU threshold
- Differentiable forward inference
- Detection parsing
- Standardized detection representation

The parser converts detector outputs into a consistent format suitable for optimization and evaluation.

---

## Validation

Successfully verified

- Model loading
- Forward inference
- Detection parsing
- Output consistency
- Device compatibility

Example output

```text
============================================================
Loading YOLO Model
============================================================

Weights

yolov8n.pt

Detector Loaded Successfully

============================================================
```

---

## Outcome

The framework gained a detector-independent interface while maintaining compatibility with YOLOv8.

---

# Commit 8 — Loss Function Framework

## Commit Message

```text
Implement differentiable loss function framework
```

---

## Objective

Introduce differentiable loss functions required for adversarial patch optimization.

The implementation was designed to support multiple optimization objectives while remaining modular.

---

## Implemented

Created

```text
attack/losses.py
```

Implemented

- Baseline optimization loss
- Confidence loss
- Objectness loss
- Total Variation (TV) loss
- Non-Printability Score (NPS) placeholder

The framework was designed so additional losses could be integrated without modifying the trainer.

---

## Validation

Verified

- Forward computation
- Gradient propagation
- Numerical stability
- Compatibility with PyTorch autograd

Example output

```text
============================================================
Testing Loss Functions
============================================================

Confidence Loss

0.7933

Objectness Loss

0.8750

TV Loss

0.6679

NPS

0.0000

============================================================
```

---

## Outcome

A modular optimization framework was established, providing the foundation for detector-aware adversarial objectives.

---

# Commit 9 — Baseline End-to-End Experiment

## Commit Message

```text
Implement baseline adversarial patch experiment
```

---

## Objective

Connect all previously implemented modules into the first executable adversarial patch pipeline.

This milestone focused on verifying that every component interacted correctly before introducing gradient-based optimization.

---

## Implemented

Integrated

- Configuration loading
- Dataset loading
- Patch initialization
- Patch application
- YOLO inference
- Detection parsing
- Confidence analysis
- Visualization generation

Produced the first complete execution pipeline from image loading through detector inference.

---

## Validation

Successfully verified

- Configuration loading
- Dataset compatibility
- Patch application
- Detector inference
- Detection parsing
- Visualization generation

Generated

- Patched image
- Detection summaries
- Confidence statistics

---

## Outcome

The first complete inference pipeline became operational, confirming that the project's modular architecture functioned correctly as an integrated system.

---

# Commit 10 — Differentiable Optimization Pipeline

## Commit Message

```text
Implement differentiable adversarial patch optimization pipeline
```

---

## Objective

Transform the inference-only pipeline into a fully differentiable optimization framework capable of learning an adversarial patch through gradient descent.

This milestone marked the transition from a static prototype to a trainable research framework.

---

## Implemented

Created

```text
attack/trainer.py

attack/attack_target.py
```

Enhanced

```text
attack/detector.py

attack/losses.py
```

Implemented

- End-to-end computational graph
- Differentiable YOLO forward pass
- Attack target abstraction
- Baseline optimization objective
- Gradient propagation
- Optimizer integration
- Patch parameter updates

The adversarial patch became the only trainable component while detector parameters remained frozen.

---

## Training Pipeline

```text
Input Image

↓

Adversarial Patch

↓

Patch Application

↓

Patched Image

↓

YOLOv8 Forward Pass

↓

Attack Target Extraction

↓

Baseline Loss

↓

Backpropagation

↓

Optimizer

↓

Updated Patch
```

---

## Validation

Successfully verified

- Differentiable forward inference
- Gradient computation
- Backpropagation
- Optimizer updates
- Patch parameter optimization

Example output

```text
============================================================
Patch Trainer
============================================================

Forward Pass Successful

Prediction Shape

torch.Size([1, 84, 8400])

Loss

9.6432

Gradient Exists

True

Gradient Shape

torch.Size([3, 160, 160])

Optimizer Step Completed

============================================================
```

---

## Outcome

This milestone established the first fully differentiable adversarial patch optimization pipeline, forming the core foundation of the research framework.

---

# Progress Summary

| Commit | Milestone | Status |
|---------|-----------|--------|
| 6 | Differentiable Patch Application | ✅ Complete |
| 7 | YOLOv8 Detector Wrapper | ✅ Complete |
| 8 | Loss Function Framework | ✅ Complete |
| 9 | Baseline End-to-End Experiment | ✅ Complete |
| 10 | Differentiable Optimization Pipeline | ✅ Complete |

At the completion of Commit 10, the project had evolved from a collection of independent modules into a functioning end-to-end adversarial patch optimization framework. The repository now supported differentiable training, enabling future work on detector-aware objectives, robust transformations, and comprehensive evaluation.

# Commit 11 — Multi-Epoch Training Infrastructure

## Commit Message

```text
Implement reusable multi-epoch training framework
```

---

## Objective

Transform the single-step optimization prototype into a reusable training framework capable of supporting long-running adversarial patch optimization experiments.

The goal was to introduce a scalable training loop while preserving the modular architecture established in previous commits.

---

## Implemented

Extended

```text
attack/trainer.py
```

Implemented

- Multi-epoch optimization
- DataLoader integration
- Batch-wise training
- Average epoch loss computation
- Patch statistics logging
- Checkpoint saving
- Loss history recording
- Configurable training parameters

Training is now entirely driven through the configuration file.

---

## Validation

Successfully trained the framework for multiple epochs.

Observed a consistent decrease in optimization loss.

Example

```text
Epoch 1

Loss

9.6432

↓

Epoch 20

Loss

9.2688
```

Successfully verified

- Stable optimization
- Gradient propagation
- Batch processing
- Checkpoint generation
- Loss logging

---

## Outcome

The project evolved from a proof-of-concept optimization loop into a reusable training engine suitable for long-running research experiments.

---

# Commit 12 — Detector-Aware Person Suppression Objective

## Commit Message

```text
Implement detector-aware person suppression loss
```

---

## Objective

Replace the placeholder optimization objective with a detector-aware loss capable of directly suppressing person detections.

This milestone marked the transition from generic optimization toward a task-specific adversarial attack.

---

## Implemented

Enhanced

```text
attack/attack_target.py

attack/losses.py

attack/trainer.py
```

Implemented

- Person detection extraction
- Person suppression loss
- Detector-aware optimization
- Target confidence statistics
- Patch clamping after optimization
- Improved checkpoint handling

The optimization objective now minimizes detector confidence specifically for the **person** class.

---

## Validation

Successfully verified

- Correct target extraction
- Person confidence computation
- Stable optimization
- Patch updates
- Gradient propagation

Observed successful reduction of person confidence during optimization.

---

## Outcome

The framework now optimized a meaningful detector-aware objective instead of a generic placeholder loss.

---

# Commit 13 — Expectation over Transformation (EOT)

## Commit Message

```text
Implement Expectation over Transformation (EOT)
```

---

## Objective

Improve adversarial patch robustness by optimizing over multiple transformed versions of the patch rather than a single static appearance.

---

## Implemented

Created

```text
attack/eot.py
```

Integrated EOT into the training pipeline.

Implemented

- Rotation transformation
- Scaling transformation
- Configuration-driven augmentation
- Differentiable transformations

Training now performs optimization over randomly transformed patches.

---

## Current Transformations

Implemented

- Rotation
- Scaling

The architecture also prepares the framework for future transformations such as

- Brightness
- Contrast
- Perspective
- Motion blur
- Gaussian noise

---

## Validation

Successfully verified

- Rotation augmentation
- Scaling augmentation
- Gradient preservation
- Stable optimization

Training completed without disrupting gradient propagation.

---

## Outcome

The framework gained improved robustness through transformation-aware optimization while preserving end-to-end differentiability.

---

# Commit 14 — Random Patch Placement

## Commit Message

```text
Implement random patch placement
```

---

## Objective

Improve patch generalization by introducing dynamic patch placement during optimization.

Instead of always placing the patch at the same coordinates, every training iteration now samples a new valid position.

---

## Implemented

Enhanced

```text
attack/trainer.py
```

Implemented

- Random coordinate generation
- Boundary validation
- Dynamic patch placement
- Per-iteration position updates

The placement algorithm ensures that the adversarial patch always remains inside the image boundaries.

---

## Validation

Successfully verified

- Random coordinates every iteration
- Boundary-safe placement
- Stable optimization
- Correct tensor dimensions

Example

```text
Patch Position

(168, 54)

↓

Patch Position

(319, 263)

↓

Patch Position

(143, 402)
```

---

## Outcome

Dynamic placement significantly improved the diversity of training samples and prepared the framework for more advanced placement strategies.

---

# Commit 15 — Confidence-Weighted Person Suppression

## Commit Message

```text
Implement confidence-weighted suppression objective
```

---

## Objective

Improve optimization by assigning greater importance to high-confidence person detections.

Instead of treating every detection equally, the optimization objective now emphasizes stronger detector predictions.

---

## Implemented

Enhanced

```text
attack/losses.py

attack/trainer.py
```

Implemented

- Confidence-weighted suppression
- Detector-aware optimization
- Stable gradient propagation
- Configurable suppression objective

The suppression loss was redesigned to prioritize difficult detections while reducing the influence of weak predictions.

---

## Validation

Successfully verified

- Stable optimization
- Reduced detector confidence
- Correct gradient computation
- Improved optimization behavior

Example training output

```text
Suppression Loss

0.353423

Gradient Exists

True

Gradient Shape

torch.Size([3, 160, 160])

Optimizer Step Completed
```

Training remained stable while producing stronger suppression behavior.

---

## Outcome

The optimization objective became significantly more effective by focusing learning on high-confidence person detections, providing a stronger foundation for future attack improvements.

---

# Progress Summary

| Commit | Milestone | Status |
|---------|-----------|--------|
| 11 | Multi-Epoch Training Infrastructure | ✅ Complete |
| 12 | Detector-Aware Person Suppression | ✅ Complete |
| 13 | Expectation over Transformation (EOT) | ✅ Complete |
| 14 | Random Patch Placement | ✅ Complete |
| 15 | Confidence-Weighted Suppression | ✅ Complete |

Following Commit 15, the framework had matured into a robust adversarial patch training system featuring configurable multi-epoch optimization, detector-aware objectives, transformation-based robustness, dynamic patch placement, and confidence-weighted suppression. These capabilities established a strong foundation for subsequent work on target selection, evaluation, adaptive placement, and comprehensive experimentation.

# Commit 16 — Top-K Target Selection

## Commit Message

```text
Implement Top-K target selection for detector-aware optimization
```

---

## Objective

Improve optimization efficiency by restricting the adversarial objective to the highest-confidence person detections.

Rather than optimizing every detection produced by YOLOv8, the framework now focuses only on the most important predictions, reducing optimization noise and improving gradient quality.

---

## Implemented

Enhanced

```text
attack/attack_target.py
```

Added

- Configurable Top-K target extraction
- Batch-wise confidence sorting
- Detector-independent target interface
- Configuration-driven Top-K selection

The implementation allows the number of optimization targets to be changed directly through the configuration file.

Example

```yaml
attack:

  top_k: 50
```

---

## Validation

Successfully verified

- Correct confidence sorting
- Stable tensor dimensions
- Batch compatibility
- Consistent gradient propagation

Example output

```text
Target Scores Shape

torch.Size([4, 50])
```

The optimization pipeline continued to train successfully using only the selected detections.

---

## Outcome

Top-K selection reduced unnecessary optimization while improving stability and preparing the framework for future detector-independent attack objectives.

---

# Commit 17 — Evaluation Framework

## Commit Message

```text
Implement adversarial patch evaluation framework
```

---

## Objective

Develop a comprehensive evaluation pipeline capable of comparing detector performance before and after adversarial patch application.

Unlike the training pipeline, which focuses on optimization, the evaluation framework provides quantitative measurements of attack effectiveness.

---

## Implemented

Created

```text
evaluation/evaluate_patch.py

evaluation/visualization.py
```

Implemented

- Original image inference
- Patched image inference
- Detection comparison
- Side-by-side visualization
- Detection statistics
- Annotated image generation

The evaluation framework was designed to operate independently from training.

---

## Validation

Successfully generated

- Original detection visualization
- Patched detection visualization
- Side-by-side comparison image

Example outputs

```text
outputs/evaluation/

original_detection.jpg

patched_detection.jpg

comparison.jpg
```

---

## Outcome

The framework gained its first complete evaluation pipeline capable of visually comparing detector performance before and after adversarial patch application.

---

# Commit 18 — Evaluation Metrics and Result Export

## Commit Message

```text
Implement evaluation metrics and export pipeline
```

---

## Objective

Extend the evaluation framework with quantitative metrics and automatic result export.

The goal was to enable systematic comparison of different experiments through reproducible evaluation statistics.

---

## Implemented

Created

```text
evaluation/metrics.py

evaluation/export.py
```

Enhanced

```text
evaluation/evaluate_patch.py
```

Implemented

- Suppression rate
- Confidence drop
- Detection retention
- CSV export
- JSON export
- Dataset evaluation summary

Evaluation results are now automatically stored for later analysis.

---

## Validation

Successfully generated

```text
results.csv

results.json
```

Example output

```text
Original Persons

3

Patched Persons

2

Suppression Rate

33.33%

Confidence Drop

21.48%

Retention

66.67%
```

---

## Outcome

The evaluation framework evolved from visual inspection to quantitative benchmarking, enabling reproducible comparison across multiple experiments.

---

# Commit 19 — Configurable Patch Initialization

## Commit Message

```text
Implement configurable patch initialization strategies
```

---

## Objective

Introduce multiple initialization strategies for the universal adversarial patch to support optimization experiments.

Different initialization methods influence convergence behavior and optimization stability.

---

## Implemented

Enhanced

```text
attack/patch.py
```

Implemented

- Gray initialization
- Random initialization
- Checkerboard initialization
- Gaussian initialization

Improved

- Validation
- Error handling
- Modular initialization logic

Initialization mode is now selected directly through the configuration file.

---

## Validation

Successfully verified all initialization strategies.

Example statistics

Gray

```text
Mean

0.5000

Std

0.0000
```

Random

```text
Mean

0.5018

Std

0.2887
```

Checkerboard

```text
Mean

0.5000

Std

0.5000
```

Gaussian

```text
Mean

0.4640

Std

0.1137
```

---

## Outcome

The patch module became significantly more flexible, enabling systematic comparison of initialization strategies during optimization.

---

# Commit 20 — Cosine Learning Rate Scheduler

## Commit Message

```text
Implement cosine learning rate scheduling
```

---

## Objective

Improve optimization stability through dynamic learning-rate scheduling.

A cosine annealing schedule was introduced to provide smooth learning-rate decay throughout training.

---

## Implemented

Enhanced

```text
attack/trainer.py

train.py

attack/configs/default.yaml
```

Implemented

- CosineAnnealingLR integration
- Scheduler configuration
- Epoch-wise learning-rate updates
- Learning-rate logging

Training now supports configurable scheduling without modifying the trainer.

Example configuration

```yaml
scheduler:

  enabled: true

  type: cosine

  T_max: 50

  eta_min: 0.0001
```

---

## Validation

Successfully verified cosine decay.

Example

```text
Epoch 1

Learning Rate

0.030000

↓

Epoch 25

Learning Rate

0.015000

↓

Epoch 50

Learning Rate

0.000100
```

Training completed successfully while preserving

- Stable gradients
- Checkpoint generation
- Loss history logging
- Patch optimization

---

## Outcome

The optimization engine gained improved convergence characteristics through smooth learning-rate scheduling, completing the first major version of the adversarial patch training framework.

---

# Progress Summary

| Commit | Milestone | Status |
|---------|-----------|--------|
| 16 | Top-K Target Selection | ✅ Complete |
| 17 | Evaluation Framework | ✅ Complete |
| 18 | Evaluation Metrics & Export | ✅ Complete |
| 19 | Configurable Patch Initialization | ✅ Complete |
| 20 | Cosine Learning Rate Scheduler | ✅ Complete |

By the completion of Commit 20, the framework had evolved into a fully functional research prototype featuring detector-aware optimization, transformation-based robustness, configurable initialization strategies, comprehensive evaluation metrics, automatic result export, and stable multi-epoch optimization. These milestones established the foundation for the next phase of development, which focused on improving checkpoint management, adaptive patch placement, composite loss design, and more advanced adversarial attack strategies.

# Commit 21 — Best Checkpoint Management

## Commit Message

```text
Implement automatic best checkpoint selection
```

---

## Objective

Improve the training pipeline by automatically preserving the best-performing adversarial patch instead of saving only the latest optimization state.

This allows evaluation to consistently use the strongest available model while preventing good checkpoints from being overwritten by later epochs.

---

## Implemented

Enhanced

```text
attack/trainer.py
```

Implemented

- Best loss tracking
- Automatic checkpoint comparison
- Best checkpoint serialization
- Training summary improvements

The trainer now stores

```text
outputs/checkpoints/

best.pt

latest.pt
```

independently.

---

## Validation

Successfully verified

- Best loss detection
- Checkpoint replacement
- Stable serialization
- Correct epoch tracking

Example output

```text
============================================================

✓ New Best Checkpoint Saved

Epoch : 2

Best Loss : 0.371475

Saved To :

outputs/checkpoints/best.pt

============================================================
```

---

## Outcome

The framework now automatically preserves the strongest optimization result, improving experiment reproducibility and evaluation consistency.

---

# Commit 22 — Bounding Box Utilities

## Commit Message

```text
Implement reusable bounding box utilities
```

---

## Objective

Separate bounding-box operations from the evaluation pipeline into reusable helper functions.

The goal was to simplify future development while supporting person-aware patch placement.

---

## Implemented

Created

```text
attack/bbox_utils.py
```

Implemented

- Person detection extraction
- Largest person selection
- Bounding-box area computation
- Bounding-box validation

These utilities are now reused by both the trainer and evaluation modules.

---

## Validation

Successfully verified

```text
============================================================

Bounding Box Utilities

============================================================

Module Loaded Successfully

============================================================
```

Confirmed

- Correct extraction
- Correct largest-box selection
- Stable tensor handling

---

## Outcome

Bounding-box processing became modular and reusable, eliminating duplicated logic and preparing the framework for adaptive patch placement.

---

# Commit 23 — Person-Aware Patch Placement

## Commit Message

```text
Implement person-aware adaptive patch placement
```

---

## Objective

Replace purely random placement with a semantic placement strategy that positions the patch relative to detected people.

This improves realism and better reflects physical-world attack scenarios.

---

## Implemented

Created

```text
attack/placement.py
```

Integrated

```text
attack/bbox_utils.py
```

Supported placement modes

- Fixed
- Random
- Person-aware

Person-aware placement automatically falls back to center placement when no person is detected.

---

## Validation

Successfully verified

```text
Largest Person Box

[12.22, 5.30, 631.26, 637.78]

Patch Position

(241,178)
```

Also verified fallback behavior

```text
Largest Person Box

None

Patch Position

(240,240)
```

---

## Outcome

Patch placement became detector-aware, significantly improving the realism of adversarial patch optimization.

---

# Commit 24 — Body Anchor Placement

## Commit Message

```text
Implement body-anchor patch placement
```

---

## Objective

Extend person-aware placement by allowing patches to be positioned relative to specific body regions.

This enables systematic evaluation of which body location produces the strongest adversarial effect.

---

## Implemented

Enhanced

```text
attack/placement.py
```

Added anchor modes

- Center
- Head
- Torso
- Feet

Placement is now entirely configuration-driven.

Example

```yaml
placement:

  mode: person

  anchor: torso
```

---

## Validation

Evaluated all supported anchors.

Observed unique patch positions for

- Head
- Torso
- Center
- Feet

Example

```text
Head

Patch Position

(241,51)
```

```text
Feet

Patch Position

(0,269)
```

---

## Outcome

The framework gained support for body-region-specific adversarial optimization, enabling systematic placement experiments.

---

# Commit 25 — Composite Loss Framework

## Commit Message

```text
Implement configurable composite loss framework
```

---

## Objective

Replace the fixed optimization objective with a configurable weighted composite loss.

This architecture allows future adversarial objectives to be added without modifying the training engine.

---

## Implemented

Enhanced

```text
attack/losses.py

attack/trainer.py
```

Implemented

- Configurable suppression weight
- TV loss weight
- NPS weight
- Composite loss computation
- Loss weight logging

Optimization now follows

```text
Total Loss

=

Suppression

+

TV

+

NPS
```

---

## Validation

Successfully verified

```text
Suppression Loss

0.346832

TV Loss

0.238005

NPS Loss

0.269162

Total Loss

0.349481
```

Confirmed

- Correct weighting
- Stable optimization
- Gradient propagation

---

## Outcome

Loss computation became significantly more flexible, enabling rapid experimentation through configuration changes alone.

---

# Commit 26 — Non-Printability Score (NPS)

## Commit Message

```text
Implement printable color regularization
```

---

## Objective

Replace the placeholder Non-Printability Score with an actual printable-color regularization objective.

This prepares the framework for future physical-world adversarial attacks.

---

## Implemented

Enhanced

```text
attack/losses.py
```

Implemented

- Printable color palette
- Distance-based NPS computation
- Differentiable regularization
- Integration into composite loss

The NPS component now contributes directly to optimization.

---

## Validation

Successfully verified

```text
============================================================

Testing Loss Functions

============================================================

NPS

0.300075

============================================================
```

Training also confirmed

```text
NPS Loss

0.257664

Total Loss

0.339975
```

without disrupting optimization.

---

## Outcome

The optimization framework now incorporates printable-color constraints, improving the foundation for future physical-world adversarial patch research.

---

# Progress Summary

| Commit | Milestone | Status |
|---------|-----------|--------|
| 21 | Best Checkpoint Management | ✅ Complete |
| 22 | Bounding Box Utilities | ✅ Complete |
| 23 | Person-Aware Patch Placement | ✅ Complete |
| 24 | Body Anchor Placement | ✅ Complete |
| 25 | Composite Loss Framework | ✅ Complete |
| 26 | Printable Color Regularization (NPS) | ✅ Complete |

By the completion of Commit 26, the framework had progressed beyond a basic research prototype into a significantly more mature adversarial attack system. Major improvements included intelligent checkpoint management, modular bounding-box processing, adaptive person-aware patch placement, configurable body-anchor experiments, a flexible composite loss architecture, and printable-color regularization. These enhancements established the groundwork for the final development phase, which focused on suppression parameterization, advanced experimentation, and research-grade evaluation.

# Commit 27 — Placement Strategy Evaluation

## Commit Message

```text
Implement placement strategy benchmarking
```

---

## Objective

Extend the evaluation framework to support systematic comparison of multiple patch placement strategies.

The goal was to determine how patch location influences adversarial attack effectiveness.

---

## Implemented

Enhanced

```text
attack/placement.py

evaluation/evaluate_patch.py
```

Implemented evaluation support for

- Fixed placement
- Random placement
- Person-aware placement

Person-aware placement supports

- Center
- Head
- Torso
- Feet

Each strategy can now be evaluated independently through configuration.

---

## Validation

Performed evaluation using multiple placement modes.

Example observations

```text
Head Placement

Attack Success Rate

22%

----------------------------

Torso Placement

Attack Success Rate

26%

----------------------------

Feet Placement

Attack Success Rate

16%

----------------------------

Random Placement

Attack Success Rate

14%
```

These experiments demonstrated that placement location has a measurable influence on attack performance.

---

## Outcome

The framework now supports reproducible placement benchmarking, enabling systematic comparison of semantic patch locations.

---

# Commit 28 — Adaptive Person Placement Refinement

## Commit Message

```text
Improve adaptive person-aware placement
```

---

## Objective

Improve the robustness and realism of person-aware placement by refining anchor computation and fallback behavior.

---

## Implemented

Enhanced

```text
attack/placement.py

attack/bbox_utils.py
```

Improved

- Anchor computation
- Boundary handling
- Bounding-box clamping
- Center fallback
- Placement consistency

Placement now remains valid even for partially visible persons.

---

## Validation

Successfully verified

- Small bounding boxes
- Large bounding boxes
- Missing detections
- Image boundary conditions

Observed stable placement across the evaluation dataset.

---

## Outcome

Adaptive placement became more reliable and suitable for larger-scale evaluation experiments.

---

# Commit 29 — Evaluation Pipeline Refinement

## Commit Message

```text
Improve evaluation framework and experiment reporting
```

---

## Objective

Enhance the evaluation pipeline by improving reporting, visualization, and experiment reproducibility.

---

## Implemented

Enhanced

```text
evaluation/evaluate_patch.py

evaluation/export.py

evaluation/metrics.py
```

Improved

- Dataset summaries
- Visualization output
- CSV export
- JSON export
- Evaluation logging

Added clearer console reporting for

- Largest detected person
- Patch placement
- Dataset statistics

---

## Validation

Successfully generated

```text
Average Suppression

14.72%

Average Confidence Drop

16.60%

Average Retention

39.28%

Attack Success Rate

20%
```

Verified

- Stable evaluation
- Correct metric aggregation
- Consistent exports

---

## Outcome

Evaluation became substantially more informative, simplifying comparison between experimental configurations.

---

# Commit 30 — Configurable Composite Loss Experiments

## Commit Message

```text
Improve configurable composite loss experimentation
```

---

## Objective

Increase flexibility of adversarial optimization by exposing composite loss parameters directly through the configuration system.

This allows rapid experimentation without modifying source code.

---

## Implemented

Enhanced

```text
attack/losses.py

attack/trainer.py

attack/configs/default.yaml
```

Added configurable

- Suppression weight
- TV weight
- NPS weight

Improved training logs by displaying

- Individual loss values
- Composite loss
- Active loss weights

---

## Validation

Successfully verified

```text
Suppression Loss

0.346832

TV Loss

0.238005

NPS Loss

0.269162

Total Loss

0.349481

Loss Weights

Suppression

1.0

TV

0.01

NPS

0.001
```

Confirmed stable optimization under different weighting schemes.

---

## Outcome

Composite loss experimentation became significantly easier, supporting future ablation studies.

---

# Commit 31 — Confidence-Weighted Suppression Experiments

## Commit Message

```text
Implement configurable confidence-weighted suppression
```

---

## Objective

Generalize the suppression objective by introducing configurable confidence weighting.

Instead of using a fixed quadratic weighting, researchers can now adjust the exponent applied to detector confidence scores.

---

## Implemented

Enhanced

```text
attack/losses.py

attack/trainer.py

attack/configs/default.yaml
```

Added

- Configurable suppression threshold
- Configurable suppression power
- Training diagnostics
- Parameter logging

Current configuration

```yaml
suppression:

  threshold: 0.25

  power: 2.0
```

---

## Validation

Compared multiple suppression exponents.

Examples

```text
Power = 1

Attack Success Rate

16%

------------------------

Power = 2

Attack Success Rate

20%

------------------------

Power = 3

Additional experimentation supported
```

Training logs now report

```text
Suppression Parameters

Threshold

0.25

Power

2.0
```

---

## Outcome

The suppression objective became fully configurable, enabling systematic investigation of confidence-weighted adversarial optimization.

---

# Current Project Status

**Framework Version**

Commit 31

---

## Completed Features

✅ Configuration-driven experiments

✅ Universal adversarial patch

✅ COCO dataset support

✅ YOLOv8 integration

✅ Differentiable optimization

✅ Person suppression objective

✅ Confidence-weighted suppression

✅ Composite configurable loss

✅ Multiple initialization strategies

✅ Random placement

✅ Person-aware placement

✅ Head / Torso / Center / Feet anchors

✅ Bounding-box utilities

✅ Expectation over Transformation

✅ Adam optimization

✅ Cosine learning-rate scheduling

✅ Multi-epoch training

✅ Best checkpoint selection

✅ Automatic checkpoint saving

✅ Loss history logging

✅ Training visualization

✅ Comprehensive evaluation

✅ Detection visualization

✅ CSV export

✅ JSON export

---

# Lessons Learned

Throughout development, several important software engineering principles proved valuable.

## Modular Design

Independent modules significantly simplified debugging and feature integration.

---

## Configuration-Driven Development

Moving experiment parameters into YAML files greatly improved reproducibility and reduced code duplication.

---

## Continuous Validation

Testing each feature immediately after implementation prevented small issues from propagating into later development stages.

---

## Incremental Development

Building the framework through small, self-contained commits resulted in a stable and maintainable codebase.

---

## Research-Oriented Engineering

Separating optimization, evaluation, visualization, and experiment management enabled rapid experimentation while preserving code quality.

---

# Next Development Milestones

Future work will focus on

- Stronger EOT transformations
- Physical-world adversarial attacks
- Multi-scale optimization
- Detector transferability
- Additional datasets
- Robustness benchmarking
- Cross-detector evaluation
- Video-based adversarial attacks

---

# Development Statistics

## Total Major Commits

31

---

## Core Modules

- Configuration
- Dataset
- Detector
- Patch
- Patch Applier
- Placement
- Bounding Box Utilities
- Attack Target
- Losses
- EOT
- Trainer
- Evaluation
- Metrics
- Visualization
- Export

---

## Software Stack

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PyYAML

---

# Final Summary

The **Robust Adversarial Patch Attack Framework for YOLOv8** has evolved from a simple project skeleton into a comprehensive, research-oriented adversarial machine learning framework through thirty-one carefully planned development milestones.

Each commit introduced a well-defined feature, validated its correctness, and established a stable foundation for subsequent work. The resulting architecture emphasizes modularity, reproducibility, extensibility, and end-to-end differentiability while supporting systematic experimentation through configuration-driven workflows.

By Commit 31, the framework provides a complete pipeline for training, evaluating, and analyzing universal adversarial patches against YOLOv8, with support for adaptive patch placement, confidence-weighted suppression, configurable composite losses, robust evaluation metrics, visualization, checkpoint management, and experiment logging.

This development log serves as a historical record of the project's engineering evolution and demonstrates the incremental design decisions that transformed the repository into a research-ready software framework suitable for future work in adversarial machine learning, computer vision, and AI security.

---

# Document Information

**Project:** Robust Adversarial Patch Attack Framework for YOLOv8

**Development Log Version:** Commit 31

**Author:** Rishab Shetty

**Institution:** PES University

**Research Areas**

- Adversarial Machine Learning
- Computer Vision
- Deep Learning
- Artificial Intelligence
- AI Security

---