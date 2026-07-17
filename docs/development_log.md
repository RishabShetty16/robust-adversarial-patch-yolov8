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