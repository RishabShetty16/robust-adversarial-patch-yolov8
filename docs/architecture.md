# Project Architecture

## Stage A Pipeline
Configuration
        │
        ▼
Dataset
        │
        ▼
Image
        │
        ▼
Patch
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
Detection Parser
        │
        ▼
Confidence Statistics

## High-Level Design

```
Configuration
        │
        ▼
Dataset ─────► Detector
        │           │
        ▼           ▼
Patch ───────► EOT Engine
        │           │
        ▼           ▼
      Loss ◄── Predictions
        │
        ▼
Optimizer
        │
        ▼
Evaluation
```

---

## attack/

Contains the complete implementation of the adversarial attack pipeline.

Modules

- config.py
- utils.py
- patch.py
- dataset.py
- detector.py
- losses.py
- eot.py
- trainer.py

---

## defense/

Defense methods against adversarial patches.

---

## physical/

Physical-world attack pipeline.

---

## evaluation/

Evaluation scripts and metrics.

---

## outputs/

Stores

- checkpoints
- figures
- logs
- optimized patches

---

## Design Principles

- Modular
- Reproducible
- Extensible
- Research-oriented