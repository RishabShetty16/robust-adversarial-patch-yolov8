# Development Log

This document tracks the progress of the project as it evolves from an initial repository skeleton into a complete research-oriented framework for robust physical adversarial patch attacks against YOLOv8.

---

# Commit 1 – Initialize Project Structure

**Commit Message**

```
Initialize project structure
```

## Objective

Create a clean, modular repository structure before implementing any functionality.

## Implemented

- Created the project repository.
- Initialized Git version control.
- Created the following top-level folders:
  - attack/
  - defense/
  - evaluation/
  - physical/
  - data/
  - outputs/
  - notebooks/
  - tests/
  - assets/
  - docs/
- Created output subdirectories:
  - checkpoints/
  - patches/
  - figures/
  - logs/
- Added:
  - README.md
  - requirements.txt
  - train.py
  - evaluate.py
- Added `.gitignore` for:
  - virtual environments
  - checkpoints
  - logs
  - generated images
  - Python cache files
- Created the initial attack module skeleton:
  - attack/__init__.py
  - attack/config.py
  - attack/utils.py
  - attack/patch.py
  - attack/dataset.py
  - attack/detector.py
  - attack/configs/default.yaml

## Status

✅ Repository successfully initialized.

---

# Commit 2 – Configuration Management

**Commit Message**

```
Implement configuration management module
```

## Objective

Implement a centralized configuration system so that all experiment parameters are loaded from YAML instead of being hardcoded.

## Implemented

### attack/config.py

Implemented:

- YAML configuration loader
- Configuration validation
- Automatic device selection
- Experiment configuration summary

### Functions

- `load_config()`
- `validate_config()`
- `get_device()`
- `print_config_summary()`

### Features

- Reads configuration from `attack/configs/default.yaml`
- Validates required configuration sections
- Automatically selects CPU or CUDA
- Prints experiment information before training

## Testing

Created a temporary test script.

Successfully verified:

- Configuration loading
- YAML parsing
- Device selection
- Configuration summary output

Output:

```
============================================================
Experiment Configuration
============================================================
Experiment : baseline_patch_attack
Device     : auto
Model      : yolov8n.pt
Patch Size : 160
Optimizer  : AdamW
LR         : 0.03
Iterations : 1000
Attack     : suppression
============================================================
cpu
```

## Status

✅ Tested successfully.

# Commit 3

## Utility Module

### Implemented

- Random seed initialization
- Output directory creation
- Timestamp generation
- Model parameter counting

Files

- attack/utils.py

Status

✅ Tested

---

# Commit 4

## Adversarial Patch Module

### Objective

Implement a reusable adversarial patch module that encapsulates all patch-related functionality.

### Implemented

- Learnable adversarial patch (`nn.Module`)
- Gray initialization
- Random initialization
- Checkerboard initialization
- Pixel clamping
- Patch statistics
- Patch save/load
- Patch visualization

### Files

- attack/patch.py

### Testing

Successfully verified:

- Patch initialization
- Statistics computation
- Patch saving
- Patch visualization

### Status

✅ Completed

---

---

# Commit 5

## COCO Dataset Module

### Objective

Build a reusable dataset loader for adversarial patch training.

### Implemented

- COCO dataset loader
- Automatic image discovery
- Image preprocessing
- Resize transformation
- Tensor conversion
- Configuration-driven paths

### Files

- attack/dataset.py
- test_dataset.py

### Testing

Verified:

- Dataset initialization
- Image loading
- Tensor conversion
- Correct tensor shape

Status

✅ Completed

---

# Commit 6

## Patch Applier Module

### Objective

Implement a reusable module that applies adversarial patches to images.

### Implemented

- Apply patch to one image
- Apply patch to image batches
- Automatic boundary handling
- Pixel value clamping

### Files

- attack/patch_applier.py

### Testing

Verified:

- Single-image patch application
- Batch patch application
- Output tensor shape
- Pixel value range

Status

✅ Completed

---

# Commit 7

## YOLO Detector Wrapper & Detection Parser

### Objective

Build a detector interface independent of Ultralytics internals.

### Implemented

- YOLO model wrapper
- Automatic model loading
- Configurable confidence threshold
- Configurable IoU threshold
- Detection parser
- Standardized detection dictionaries

### Files

- attack/detector.py
- attack/parser.py

### Testing

Verified:

- YOLO loads correctly
- Inference executes successfully
- Detections parsed correctly

Status

✅ Completed

---

# Commit 8

## Individual Loss Functions

### Objective

Implement differentiable loss functions for adversarial patch optimization.

### Implemented

- Confidence loss
- Objectness loss
- Total Variation (TV) loss
- NPS placeholder

### Files

- attack/losses.py

### Testing

Verified:

- Confidence loss
- Objectness loss
- TV loss
- Gradient compatibility

Status

✅ Completed

# Commit 9

## Stage A Baseline Experiment

### Objective

Create the first executable experiment connecting all completed modules.

### Implemented

- Configuration loading
- Dataset loading
- Patch creation
- Optimizer creation
- Patch application
- Patched image visualization
- YOLO detection
- Detection parsing
- Confidence statistics

### Outputs

- Patched image
- Detection list
- Confidence summary

### Status

✅ Completed

---

# Commit 10

## Differentiable Adversarial Patch Optimization Pipeline

### Commit Message

```text
Implement differentiable adversarial patch optimization pipeline
```

### Objective

Transform the baseline inference pipeline into a fully differentiable optimization framework capable of updating the adversarial patch using gradient descent.

### Implemented

- Implemented differentiable YOLO forward pass
- Added Attack Target abstraction
- Integrated baseline optimization loss
- Built end-to-end computational graph
- Enabled gradient propagation from detector to adversarial patch
- Integrated optimizer update
- Verified gradient flow through the complete pipeline

### Files

- attack/trainer.py
- attack/attack_target.py
- attack/losses.py
- attack/detector.py
- experiments/stage_a_train.py

### Training Pipeline

```
Image
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
YOLO Forward Pass
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

### Testing

Verified:

- Differentiable forward pass
- Loss computation
- Backpropagation
- Optimizer update
- Gradient propagation to the adversarial patch

Example Output

```
============================================================
Patch Trainer
============================================================

Forward Pass Successful

Prediction Shape :
torch.Size([1, 84, 8400])

Loss :
9.64319896697998

Gradient Exists :
True

Gradient Shape :
torch.Size([3, 160, 160])

Optimizer Step Completed
============================================================
```

### Status

✅ Completed

# Commit 11

## Training Infrastructure

### Objective

Transform the differentiable optimization pipeline into a reusable training framework supporting long-running experiments.

### Added

- Configurable multi-epoch training
- DataLoader integration
- Modular training loop
- Batch processing
- Average epoch loss computation
- Checkpoint saving
- Loss history logging
- Patch statistics logging

### Validation

Successfully trained for 20 epochs.

Observed monotonic reduction in optimization loss:

Epoch 1
Loss = 9.6432

↓

Epoch 20
Loss = 9.2688

The successful decrease confirms that gradients propagate correctly from the detector output back to the adversarial patch while maintaining stable optimization.

### Next

Implement a detector-specific adversarial objective (person suppression loss) to replace the baseline optimization loss.

# Commit 12

## Goal

Replace the placeholder optimization objective with a detector-aware suppression loss.

## Completed

✓ Implemented AttackTarget for parsing YOLO outputs.

✓ Added person_suppression_loss().

✓ Integrated detector-aware loss into the trainer.

✓ Added target confidence statistics.

✓ Verified gradient flow.

✓ Added patch clamping after every optimizer step.

✓ Added checkpoint saving.

✓ Added loss history logging.

## Validation

- Detector outputs successfully parsed.
- Target confidence extracted correctly.
- Loss decreases during training.
- Patch updates successfully.
- Patch values remain within [0,1].
- Training completes successfully.

## Status

Stable training pipeline achieved.

---

# Commit 13

## Expectation over Transformation (EOT)

### Objective

Improve the robustness of adversarial patch optimization by introducing Expectation over Transformation (EOT), allowing the patch to be optimized under varying image transformations.

### Implemented

- Added EOT module (`attack/eot.py`)
- Rotation transformation
- Scaling transformation
- Configuration-driven augmentation
- Differentiable transformation pipeline

### Files

- attack/eot.py
- attack/configs/default.yaml
- attack/trainer.py

### Validation

Verified:

- Rotation applied successfully
- Scaling applied successfully
- Gradient propagation preserved
- Training remained stable

### Status

✅ Completed

---

# Commit 14

## Random Patch Placement

### Objective

Improve attack generalization by placing the adversarial patch at random valid locations during training instead of using a fixed position.

### Implemented

- Random patch position generator
- Automatic boundary checking
- Dynamic patch placement every training iteration

### Files

- attack/trainer.py

### Validation

Verified:

- Random coordinates generated every iteration
- Patch remained within image boundaries
- Training completed successfully

### Status

✅ Completed

---

# Commit 15

## Confidence-Weighted Person Suppression

### Objective

Improve optimization by focusing on the most confident person detections instead of treating all detections equally.

### Implemented

- Confidence-weighted suppression objective
- Improved detector-aware optimization
- Stable gradient propagation

### Files

- attack/losses.py
- attack/trainer.py

### Validation

Verified:

- Stable optimization
- Reduced person confidence
- Successful gradient computation

### Status

✅ Completed

---

# Commit 16

## Top-K Target Selection

### Objective

Restrict optimization to the highest-confidence detections, improving efficiency and stability.

### Implemented

- Configurable Top-K target extraction
- Batch-wise confidence selection
- Detector-independent target interface

### Files

- attack/attack_target.py
- attack/configs/default.yaml

### Validation

Verified:

- Correct Top-K extraction
- Stable tensor shapes
- Successful optimization

### Status

✅ Completed

---

# Commit 17

## Evaluation Framework

### Objective

Develop a comprehensive evaluation framework for comparing detector performance before and after adversarial patch application.

### Implemented

- Original image inference
- Patched image inference
- Detection comparison
- Detection visualization
- Side-by-side comparison generation

### Added

- Evaluation metrics
- Detection statistics
- Annotated outputs

### Files

- evaluation/evaluate_patch.py
- evaluation/visualization.py

### Validation

Successfully generated:

- Original detection visualization
- Patched detection visualization
- Comparison image

### Status

✅ Completed

---

# Commit 18

## Enhanced Evaluation Metrics and Export

### Objective

Extend the evaluation pipeline with quantitative metrics and export functionality.

### Implemented

- Detection suppression rate
- Confidence drop metric
- Detection retention metric
- CSV export
- JSON export
- Evaluation summary printing

### Files

- evaluation/metrics.py
- evaluation/export.py
- evaluation/evaluate_patch.py

### Validation

Successfully exported:

- results.csv
- results.json

Example output:

Original Persons : 3

Patched Persons : 3

Suppression Rate : 0%

Confidence Drop : 0%

Retention Rate : 100%

### Status

✅ Completed

---

# Commit 19

## Configurable Patch Initialization

### Objective

Provide multiple initialization strategies for the learnable adversarial patch to support experimentation.

### Implemented

Added four initialization strategies:

- Gray
- Random
- Checkerboard
- Gaussian

Implemented modular initialization functions.

Improved validation and error handling.

### Files

- attack/patch.py

### Validation

Verified all initialization modes.

Example statistics:

Gray

Mean : 0.5000

Std : 0.0000

Random

Mean : 0.5018

Std : 0.2887

Checkerboard

Mean : 0.5000

Std : 0.5000

Gaussian

Mean : 0.4640

Std : 0.1137

### Status

✅ Completed

---

# Commit 20

## Cosine Learning Rate Scheduler

### Objective

Improve optimization stability by introducing learning rate scheduling during adversarial patch training.

### Implemented

- Configurable scheduler section in YAML
- CosineAnnealingLR integration
- Scheduler support inside the training engine
- Learning rate logging after every epoch

### Files

- train.py
- attack/trainer.py
- attack/configs/default.yaml

### Validation

Successfully verified cosine decay:

Epoch 1

Learning Rate : 0.030000

↓

Epoch 25

Learning Rate : 0.015000

↓

Epoch 50

Learning Rate : 0.000100

Training completed successfully while preserving:

- Gradient propagation
- Checkpoint saving
- Loss history logging

### Status

✅ Completed

---

# Current Progress

## Completed

- Configuration Management
- Utility Module
- Adversarial Patch Module
- COCO Dataset Loader
- Patch Application
- YOLOv8 Detector Wrapper
- Detection Parser
- Loss Function Framework
- Differentiable Optimization Pipeline
- Multi-Epoch Training
- Person Suppression Objective
- Expectation over Transformation (EOT)
- Random Patch Placement
- Confidence-Weighted Optimization
- Top-K Target Selection
- Evaluation Framework
- Evaluation Metrics
- CSV / JSON Export
- Multiple Patch Initialization Strategies
- Cosine Learning Rate Scheduler

---

# Upcoming Milestones

- Commit 21 — Best Checkpoint Selection
- Commit 22 — Training Loss Visualization
- Commit 23 — Multi-Image Training
- Commit 24 — Multi-Image Evaluation
- Commit 25 — Enhanced EOT Transformations
- Commit 26 — Physical Robustness Experiments

---

# Project Status

**Current Stage:** Commit 20 — End-to-End Adversarial Patch Training Framework

The repository now provides a complete research-oriented framework for training and evaluating adversarial patches against YOLOv8. It includes configurable optimization, expectation over transformation, quantitative evaluation, multiple initialization strategies, and a modular architecture designed for future research in physical adversarial attacks.