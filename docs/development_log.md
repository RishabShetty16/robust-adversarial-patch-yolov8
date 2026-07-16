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