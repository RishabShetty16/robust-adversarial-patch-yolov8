# Robust Adversarial Patch Attack for YOLOv8

## Overview

This repository implements a modular framework for developing robust adversarial patch attacks against YOLOv8.

The project is being developed incrementally, with each milestone introducing a new component of the attack pipeline.

---

## Current Pipeline

```
Configuration
        ↓
Dataset Loader
        ↓
Adversarial Patch
        ↓
Patch Applier
        ↓
YOLOv8 Detector
        ↓
Detection Parser
        ↓
Loss Functions
```

---

## Current Status

### ✅ Completed

- Configuration Management
- Dataset Loader
- Adversarial Patch
- Patch Applier
- YOLO Detector Wrapper
- Detection Parser
- Loss Functions
- Stage A Baseline Experiment

### 🚧 In Progress

- Differentiable Optimization Loop
- Expectation over Transformation (EOT)
- Trainer
- Evaluation

---

## Repository Structure

```
attack/
experiments/
data/
outputs/
tests/
```