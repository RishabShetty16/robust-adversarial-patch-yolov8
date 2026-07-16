# Project Architecture

## Stage A Pipeline

```text
COCO Dataset
      │
      ▼
Dataset Loader
      │
      ▼
Patch Applier
      │
      ▼
EOT Transformations
      │
      ▼
YOLO Detector
      │
      ▼
Loss Functions
      │
      ▼
Optimizer
      │
      ▼
Update Patch
```

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