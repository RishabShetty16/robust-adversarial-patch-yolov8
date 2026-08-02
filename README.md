# Robust Adversarial Patch Attack Framework for YOLOv8

A modular, research-oriented framework for generating **universal adversarial patches** against **YOLOv8** object detectors using **PyTorch**.

The framework focuses on **person suppression attacks** through fully differentiable optimization, adaptive patch placement, configurable loss functions, expectation over transformation (EOT), and comprehensive evaluation pipelines.

Unlike a simple proof-of-concept implementation, this repository is designed as a **research framework** for experimenting with adversarial patch generation, optimization strategies, placement algorithms, and detector-aware attack objectives.

---

# Overview

Adversarial patches are universal perturbations that can be physically printed or digitally applied to images in order to manipulate the predictions of deep neural networks.

This project investigates robust adversarial patch generation against **YOLOv8**, with an emphasis on suppressing **person detections** while maintaining attack robustness under various transformations.

The framework provides:

- End-to-end differentiable optimization
- Universal patch learning
- Detector-aware optimization
- Adaptive person-aware patch placement
- Configurable attack objectives
- Research-oriented evaluation
- Extensive experiment support

The modular design allows researchers to easily extend the framework with new optimization objectives, transformation pipelines, datasets, placement strategies, and evaluation protocols.

---

# Key Features

## Universal Adversarial Patch

- Learnable universal adversarial patch
- End-to-end differentiable optimization
- Multiple initialization strategies
  - Gray
  - Random
  - Gaussian
  - Checkerboard
- Automatic checkpoint serialization
- Pixel value clamping

---

## YOLOv8 Integration

- Native Ultralytics YOLOv8 integration
- Forward-pass optimization
- Detection parsing
- Device auto-selection
- Configurable confidence threshold
- Configurable IoU threshold

---

## Dataset Support

- COCO Dataset
- PyTorch Dataset interface
- DataLoader integration
- Configurable image resolution
- Configurable batch size
- Multi-image optimization

---

## Adaptive Patch Placement

Supports multiple placement strategies.

### Fixed Placement

Places the patch at predefined image coordinates.

### Random Placement

Randomly samples valid patch positions.

### Person-aware Placement

Automatically detects the largest person in the scene and positions the patch relative to configurable body anchors.

Supported anchors include:

- Center
- Head
- Torso
- Feet
- Random

Automatic fallback to center placement is used when no person is detected.

---

## Expectation over Transformation (EOT)

Robust patch optimization through randomized transformations.

Current transformations:

- Rotation
- Scaling

Framework support exists for future additions such as:

- Brightness
- Contrast
- Perspective
- Motion Blur
- Gaussian Noise

---

## Composite Loss Framework

Configurable multi-objective optimization consisting of:

- Person Suppression Loss
- Total Variation (TV) Loss
- Non-Printability Score (NPS)

Each loss component can be independently weighted through the configuration file without modifying the training code.

---

## Confidence-weighted Suppression

The suppression objective supports configurable confidence weighting using an adjustable exponent.

Features include:

- Configurable confidence threshold
- Configurable suppression power
- High-confidence emphasis
- Research-oriented ablation support

---

## Training Pipeline

- Multi-epoch optimization
- Automatic checkpoint saving
- Best checkpoint selection
- Cosine learning-rate scheduling
- Adam optimizer
- Gradient-based optimization
- Patch statistics logging
- CSV loss history export
- Automatic training curve generation

---

## Evaluation Framework

Comprehensive dataset-wide evaluation including:

- Original image inference
- Patched image inference
- Detection comparison
- Person suppression metrics
- Confidence statistics
- Attack Success Rate
- Detection retention
- CSV export
- JSON export
- Annotated visualizations
- Side-by-side comparison images

---

# Framework Architecture

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
                      Adaptive Patch Placement
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
          Fixed               Random            Person-aware
                                                     │
                                                     ▼
                                   Head / Torso / Feet / Center
                                                     │
                                                     ▼
                           Differentiable Patch Applier
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
                Confidence-weighted Suppression Loss
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
                           Updated Patch Weights
                                  │
          ┌───────────────────────┴────────────────────────┐
          ▼                                                ▼
   Checkpoint Saving                              Training Statistics
          │                                                │
          ▼                                                ▼
    Best Model Selection                           Loss History Logging
```

---

# Highlights

✔ Fully differentiable optimization pipeline

✔ Universal adversarial patch learning

✔ Modular research-oriented architecture

✔ Adaptive person-aware patch placement

✔ Configurable confidence-weighted suppression

✔ Composite configurable loss functions

✔ Expectation over Transformation (EOT)

✔ Automatic checkpoint management

✔ Dataset-wide evaluation

✔ Visualization and metric export

✔ Research ablation studies

✔ Easily extensible for future adversarial attack research

---

# Repository Structure

```text
robust-adversarial-patch-yolov8/

│
├── attack/
│   ├── configs/
│   │   └── default.yaml          # Global configuration
│   │
│   ├── attack_target.py          # Target selection
│   ├── bbox_utils.py             # Bounding-box utilities
│   ├── config.py                 # Configuration loader
│   ├── dataset.py                # COCO dataset loader
│   ├── detector.py               # YOLOv8 wrapper
│   ├── eot.py                    # Expectation over Transformation
│   ├── losses.py                 # Composite loss functions
│   ├── parser.py                 # Detection parser
│   ├── patch.py                  # Universal adversarial patch
│   ├── patch_applier.py          # Patch application
│   ├── placement.py              # Patch placement strategies
│   ├── trainer.py                # Training engine
│   └── utils.py                  # Helper utilities
│
├── evaluation/
│   ├── evaluate_patch.py         # Complete evaluation pipeline
│   ├── export.py                 # CSV / JSON export
│   ├── metrics.py                # Evaluation metrics
│   └── visualization.py          # Detection visualization
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
│   └── coco/
│
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/RishabShetty16/robust-adversarial-patch-yolov8.git

cd robust-adversarial-patch-yolov8
```

---

Create a virtual environment

Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

---

Install dependencies

```bash
pip install -r requirements.txt
```

---

Verify the installation

```bash
python test_config.py
```

Expected output

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

# Requirements

Recommended environment

| Package | Version |
|----------|---------|
| Python | 3.10+ |
| PyTorch | 2.x |
| TorchVision | Latest |
| Ultralytics | Latest |
| NumPy | Latest |
| OpenCV | Latest |
| PyYAML | Latest |

GPU acceleration is recommended but **not required**.

The framework supports both

- CPU training
- CUDA training

through automatic device selection.

---

# Dataset Preparation

The framework currently supports the **COCO dataset**.

Directory structure

```text
data/

└── coco/
    │
    ├── images/
    │
    └── labels/
```

Example

```text
data/

└── coco/

    ├── images/
    │   ├── 000000000001.jpg
    │   ├── 000000000002.jpg
    │   └── ...

    └── labels/
        ├── 000000000001.txt
        ├── 000000000002.txt
        └── ...
```

The dataset loader automatically

- loads images
- resizes images
- converts them into tensors
- prepares mini-batches
- interfaces with the PyTorch DataLoader

---

# Configuration System

The framework is entirely configuration-driven.

All experiment parameters are stored inside

```text
attack/configs/default.yaml
```

This enables researchers to modify experiments without changing source code.

Current configurable modules include

- Dataset
- YOLOv8 detector
- Patch initialization
- Patch size
- Optimizer
- Scheduler
- Training
- Evaluation
- Attack mode
- Patch placement
- Expectation over Transformation
- Composite loss
- Confidence-weighted suppression

---

# Example Configuration

```yaml
experiment:
  name: baseline_patch_attack

device: auto

patch:
  size: 160

optimizer:
  type: Adam
  lr: 0.03

scheduler:
  enabled: true
  type: cosine

placement:
  mode: person
  anchor: head

loss:
  suppression_weight: 1.0
  tv_weight: 0.0
  nps_weight: 0.001

  suppression:
    threshold: 0.25
    power: 2.0
```

Changing a single value inside the configuration file immediately changes the behavior of the training pipeline.

---

# Project Design Philosophy

The repository follows a modular design.

Each component is responsible for a single task.

```text
Dataset
      │
      ▼
Detector
      │
      ▼
Patch
      │
      ▼
Placement
      │
      ▼
Loss
      │
      ▼
Trainer
      │
      ▼
Evaluation
```

This architecture makes the framework easy to

- extend
- debug
- benchmark
- reproduce
- compare experiments

without modifying unrelated modules.

---

# Modular Components

The repository separates functionality into independent modules.

| Module | Responsibility |
|---------|----------------|
| `dataset.py` | COCO dataset loading |
| `detector.py` | YOLOv8 wrapper |
| `patch.py` | Universal patch optimization |
| `patch_applier.py` | Differentiable patch application |
| `placement.py` | Adaptive patch placement |
| `bbox_utils.py` | Person bounding-box utilities |
| `attack_target.py` | Detector-aware target selection |
| `losses.py` | Composite optimization losses |
| `trainer.py` | Multi-epoch training pipeline |
| `evaluate_patch.py` | Dataset-wide evaluation |
| `metrics.py` | Suppression metrics |
| `visualization.py` | Detection visualization |

Each module can be independently extended or replaced, enabling rapid experimentation with new adversarial attack strategies.

---

# Training Pipeline

The framework implements a fully differentiable adversarial patch optimization pipeline. During training, a universal adversarial patch is optimized to suppress **person detections** produced by YOLOv8 while remaining robust to geometric transformations and varying patch placements.

The complete optimization pipeline is illustrated below.

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
              Adaptive Patch Placement
                           │
                           ▼
               Differentiable Patch Applier
                           │
                           ▼
                   YOLOv8 Forward Pass
                           │
                           ▼
                 Detection Parsing Module
                           │
                           ▼
                Attack Target Selection
                           │
                           ▼
        Confidence-weighted Suppression Loss
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
                  Updated Patch Weights
                           │
          ┌────────────────┴────────────────┐
          ▼                                 ▼
Checkpoint Saving                 Training Statistics
```

---

# Universal Adversarial Patch

The adversarial patch is implemented as a **learnable PyTorch parameter**, allowing gradients to propagate directly from the detector output back to the patch pixels.

Unlike image-specific perturbations, a **single universal patch** is optimized across all training images.

Current implementation supports multiple initialization strategies:

- Gray
- Random
- Gaussian
- Checkerboard

Each optimization step performs

- Forward propagation
- Gradient computation
- Parameter update
- Pixel clamping

to ensure all pixel values remain within the valid image range.

---

# Patch Initialization

Different initialization methods influence convergence and optimization stability.

Currently implemented strategies include

| Initialization | Description |
|---------------|-------------|
| Gray | Constant gray patch |
| Random | Uniform random initialization |
| Gaussian | Normally distributed pixel values |
| Checkerboard | Structured alternating pattern |

The initialization strategy can be selected directly through the configuration file without modifying the source code.

---

# Patch Placement

The framework supports multiple placement strategies for applying the adversarial patch to an image.

## Fixed Placement

The patch is always placed at predefined image coordinates.

Suitable for:

- controlled experiments
- debugging
- reproducibility

---

## Random Placement

The patch location is randomly sampled inside the image boundaries for every optimization step.

Advantages

- improved robustness
- reduced overfitting
- better generalization

---

## Person-aware Placement

The framework automatically detects the largest person within the image and places the patch relative to configurable body anchors.

This strategy allows the optimization process to focus on semantically meaningful regions of the detected person.

Supported anchors include

- Center
- Head
- Torso
- Feet

If no person is detected, the framework automatically falls back to center placement.

---

# Placement Modes

| Mode | Description |
|------|-------------|
| Fixed | User-defined coordinates |
| Random | Random image position |
| Person | Largest detected person |

---

# Person Anchors

When using **person-aware placement**, the patch can be positioned relative to different body regions.

## Center

Places the patch at the geometric center of the detected person.

---

## Head

Places the patch near the upper region of the person.

Useful for studying attacks against facial and upper-body detections.

---

## Torso

Places the patch around the chest region.

This often provides a good balance between visibility and detector influence.

---

## Feet

Places the patch near the lower body.

Useful for evaluating robustness under alternative placement strategies.

---

# Bounding Box Processing

The framework automatically

- extracts person detections
- computes bounding-box dimensions
- identifies the largest detected person
- computes anchor coordinates
- clamps the final patch position inside image boundaries

This enables stable placement even when detections occur near image borders.

---

# Differentiable Patch Application

The adversarial patch is applied directly to image tensors using differentiable tensor operations.

Features include

- batch-wise processing
- automatic broadcasting
- gradient preservation
- boundary-safe placement

Because the application is fully differentiable, gradients flow from YOLOv8 predictions directly back into the patch parameters.

---

# Expectation over Transformation (EOT)

To improve robustness, the framework performs randomized transformations before applying the patch.

Current transformations include

- Random Rotation
- Random Scaling

These transformations encourage the learned patch to remain effective under varying viewpoints and scales.

Future extensions may include

- Brightness
- Contrast
- Gaussian Noise
- Perspective Warp
- Motion Blur
- Color Jitter

---

# Attack Target Selection

The framework performs detector-aware optimization by extracting only the detections relevant to the attack objective.

Current implementation supports

- Person class filtering
- Top-K confidence selection
- Batch-wise target extraction

Only the selected detections contribute to the optimization objective.

This reduces unnecessary optimization on unrelated object classes and improves attack efficiency.

---

# Confidence-weighted Suppression

Rather than treating every detected person equally, the framework places greater emphasis on high-confidence detections.

The suppression objective is defined as

```text
Loss = mean(confidence^power)
```

where

- **confidence** represents the detector confidence score
- **power** controls how strongly high-confidence detections are emphasized

Higher values of **power** increase the optimization pressure on the strongest detections while reducing the influence of weaker detections.

Both the confidence threshold and exponent are configurable through the experiment configuration file.

---

# Composite Loss

The overall optimization objective combines multiple loss components.

```text
Total Loss

=

Suppression Loss
+
TV Weight × Total Variation Loss
+
NPS Weight × Non-Printability Score
```

The contribution of each loss component can be adjusted independently through the configuration file.

Current components include

- Person Suppression Loss
- Total Variation Loss
- Non-Printability Score (NPS)

This modular design enables rapid experimentation with different optimization objectives.

---

# Optimization

Training uses gradient-based optimization.

Current optimizer

- Adam

Learning-rate scheduling

- Cosine Annealing

Automatic features

- Gradient computation
- Gradient clipping
- Patch clamping
- Best checkpoint selection
- Loss logging

The optimizer updates only the adversarial patch while keeping the detector weights frozen.

---

# Training Loop

Each optimization iteration follows the sequence below.

```text
Load Mini-batch
        │
        ▼
Transform Patch (EOT)
        │
        ▼
Compute Patch Position
        │
        ▼
Apply Patch
        │
        ▼
YOLOv8 Forward Pass
        │
        ▼
Extract Person Detections
        │
        ▼
Compute Composite Loss
        │
        ▼
Backpropagation
        │
        ▼
Update Patch
        │
        ▼
Clamp Pixel Values
        │
        ▼
Next Mini-batch
```

---

# Training Outputs

During optimization, the framework automatically records

- Training loss
- Learning rate
- Patch statistics
- Gradient information
- Checkpoints
- Best-performing model
- Training curves

Generated artifacts are stored inside

```text
outputs/

├── checkpoints/
├── logs/
├── figures/
└── patches/
```

---

# Evaluation Framework

The framework includes a comprehensive evaluation pipeline for measuring the effectiveness of adversarial patches on unseen images.

Rather than evaluating a single image, the framework performs **dataset-wide evaluation**, generating quantitative metrics, qualitative visualizations, and exportable reports.

The evaluation pipeline compares object detections before and after applying the adversarial patch, providing a detailed analysis of attack performance.

---

# Evaluation Pipeline

The evaluation process follows the workflow below.

```text
                 Evaluation Dataset
                        │
                        ▼
               Original Image Inference
                        │
                        ▼
                 YOLOv8 Detection
                        │
                        ▼
                Original Detection Results
                        │
                        ├─────────────────────┐
                        │                     │
                        ▼                     ▼
              Apply Adversarial Patch    Save Original
                        │
                        ▼
               Patched Image Inference
                        │
                        ▼
                 YOLOv8 Detection
                        │
                        ▼
               Patched Detection Results
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

---

# Evaluation Metrics

The framework computes multiple evaluation metrics for each image as well as aggregate dataset statistics.

---

## Average Suppression

Measures how many person detections are removed by the adversarial patch.

```text
Suppression

=

Original Persons − Patched Persons
----------------------------------
Original Persons
```

Higher values indicate stronger attacks.

---

## Confidence Drop

Measures the reduction in average confidence scores after patch application.

```text
Confidence Drop

=

Original Confidence − Patched Confidence
```

Higher confidence drop indicates that the detector becomes less certain about its predictions.

---

## Retention Rate

Measures the percentage of detections that remain after the attack.

```text
Retention

=

Patched Persons
---------------
Original Persons
```

Lower retention corresponds to stronger suppression.

---

## Attack Success Rate (ASR)

The Attack Success Rate measures the proportion of evaluated images in which the adversarial patch successfully suppresses the target object according to the configured evaluation criterion.

Higher ASR indicates a more effective adversarial attack.

---

# Evaluation Outputs

The framework automatically generates multiple outputs during evaluation.

## Visualizations

- Original detections
- Patched detections
- Side-by-side comparison images

Example directory

```text
outputs/

└── evaluation/

    ├── original_detection.jpg

    ├── patched_detection.jpg

    └── comparison.jpg
```

---

## Exported Metrics

Evaluation statistics are automatically exported as

- CSV
- JSON

Example

```text
outputs/

└── evaluation/

    ├── results.csv

    └── results.json
```

These files facilitate further analysis, plotting, and experiment comparison.

---

# Example Evaluation

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

---

# Visualization

The framework automatically produces annotated detection images.

Generated visualizations include

- Original detections
- Patched detections
- Side-by-side comparisons

These qualitative results complement the quantitative metrics and make it easier to inspect attack behavior.

---

# Experimentation

The repository is designed as a research framework and supports systematic experimentation through a configuration-driven workflow.

Experiments can be conducted by modifying the configuration file without changing the training code.

Current research experiments include

- Placement strategy comparison
- Loss ablation
- Suppression power analysis

---

# Placement Ablation Study

The framework supports multiple patch placement strategies.

Implemented placement modes include

| Strategy | Description |
|----------|-------------|
| Fixed | Predefined coordinates |
| Random | Random image locations |
| Center | Largest person center |
| Head | Largest person head |
| Torso | Largest person torso |
| Feet | Largest person feet |

These strategies were evaluated to determine the most effective patch placement for suppressing person detections.

---

# Composite Loss Ablation

The optimization objective consists of three configurable components.

- Person Suppression Loss
- Total Variation (TV) Loss
- Non-Printability Score (NPS)

Each component can be enabled, disabled, or re-weighted independently.

Example configuration

```yaml
loss:

  suppression_weight: 1.0

  tv_weight: 0.0

  nps_weight: 0.001
```

This design enables rapid experimentation with different optimization objectives.

---

# Suppression Power Ablation

The suppression loss supports configurable confidence weighting.

The optimization objective is defined as

```text
Loss

=

mean(confidence^power)
```

The exponent controls how strongly high-confidence detections influence optimization.

Evaluated configurations

| Power | Description |
|--------|-------------|
| 1.0 | Linear weighting |
| 2.0 | Quadratic weighting |
| 3.0 | Cubic weighting |

Experimental results indicated that **power = 2.0** achieved the best balance between suppression effectiveness and attack success.

---

# Current Best Configuration

The best-performing configuration obtained during experimentation is

```yaml
placement:

  mode: person

  anchor: head

loss:

  suppression_weight: 1.0

  tv_weight: 0.0

  nps_weight: 0.001

  suppression:

    threshold: 0.25

    power: 2.0
```

---

# Experimental Results

Current best evaluation results

| Metric | Value |
|---------|------:|
| Average Suppression | **15.98%** |
| Average Confidence Drop | **17.27%** |
| Average Retention | **38.02%** |
| Attack Success Rate | **26.00%** |

These results demonstrate that adaptive placement together with confidence-weighted suppression improves the effectiveness of the generated adversarial patch.

---

# Reproducibility

The framework has been designed with reproducible experimentation in mind.

All important hyperparameters are stored in the configuration file, including

- Dataset parameters
- Patch size
- Optimizer
- Learning rate
- Scheduler
- Placement strategy
- Loss weights
- Suppression parameters
- Evaluation settings

As a result, experiments can be reproduced simply by sharing the corresponding configuration file.

---

# Usage

## Training

Train the universal adversarial patch using the default configuration.

```bash
python train.py
```

During training, the framework automatically performs:

- COCO dataset loading
- Patch initialization
- Expectation over Transformation (EOT)
- Adaptive patch placement
- YOLOv8 forward propagation
- Confidence-weighted suppression optimization
- Composite loss computation
- Gradient backpropagation
- Patch parameter updates
- Checkpoint saving
- Loss history logging
- Training curve generation

---

## Evaluation

Evaluate the trained adversarial patch.

```bash
python -m evaluation.evaluate_patch
```

The evaluation pipeline automatically performs:

- Original image inference
- Patched image inference
- Detection comparison
- Metric computation
- Visualization generation
- CSV export
- JSON export

---

# Generated Outputs

After training and evaluation, the repository automatically creates the following directory structure.

```text
outputs/

├── checkpoints/
│   ├── best.pt
│   └── latest.pt
│
├── evaluation/
│   ├── comparison.jpg
│   ├── original_detection.jpg
│   ├── patched_detection.jpg
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

---

# Current Progress

| Component | Status |
|-----------|--------|
| Configuration System | ✅ Complete |
| COCO Dataset Loader | ✅ Complete |
| YOLOv8 Integration | ✅ Complete |
| Universal Patch Optimization | ✅ Complete |
| Differentiable Patch Application | ✅ Complete |
| Adaptive Patch Placement | ✅ Complete |
| Person-aware Placement | ✅ Complete |
| Bounding Box Utilities | ✅ Complete |
| Attack Target Extraction | ✅ Complete |
| Confidence-weighted Suppression | ✅ Complete |
| Composite Loss Framework | ✅ Complete |
| Loss Ablation Study | ✅ Complete |
| Suppression Power Ablation | ✅ Complete |
| EOT (Rotation) | ✅ Complete |
| EOT (Scaling) | ✅ Complete |
| Multi-Epoch Training | ✅ Complete |
| Cosine Learning Rate Scheduler | ✅ Complete |
| Automatic Checkpoint Saving | ✅ Complete |
| Best Checkpoint Selection | ✅ Complete |
| Evaluation Framework | ✅ Complete |
| Visualization Pipeline | ✅ Complete |
| CSV / JSON Export | ✅ Complete |

---

# Experimental Highlights

The framework currently supports several research-oriented experiments.

## Placement Strategies

- Fixed
- Random
- Person Center
- Person Head
- Person Torso
- Person Feet

---

## Loss Function Experiments

- Composite Loss
- TV Loss
- Non-Printability Score (NPS)
- Confidence-weighted Suppression

---

## Suppression Power Study

The suppression objective supports configurable confidence weighting.

Evaluated configurations include:

| Power | Purpose |
|--------|---------|
| 1.0 | Linear confidence weighting |
| 2.0 | Quadratic confidence weighting |
| 3.0 | Cubic confidence weighting |

Experimental evaluation indicated that **power = 2.0** provides the best balance between suppression effectiveness and attack success.

---

# Current Best Results

Best results obtained using the current framework.

| Metric | Value |
|---------|-------|
| Average Suppression | **15.98%** |
| Average Confidence Drop | **17.27%** |
| Average Retention | **38.02%** |
| Attack Success Rate | **26.00%** |

These results were obtained using:

- Person-aware placement
- Confidence-weighted suppression
- Composite loss optimization
- Expectation over Transformation (Rotation + Scaling)

---

# Research Contributions

This repository currently implements:

- Universal adversarial patch optimization
- Detector-aware attack optimization
- Adaptive person-aware patch placement
- Configurable composite loss framework
- Confidence-weighted suppression objective
- End-to-end differentiable optimization
- Dataset-wide evaluation framework
- Comprehensive experiment logging
- Configuration-driven experimentation
- Modular research architecture

The framework is designed to serve as a foundation for future research on adversarial attacks against modern object detectors.

---

# Roadmap

Future improvements include:

## Robustness

- Perspective transformation
- Brightness augmentation
- Contrast augmentation
- Motion blur
- Gaussian noise
- Weather effects

---

## Optimization

- Adaptive Top-K target selection
- Dynamic confidence thresholds
- Multi-scale optimization
- Mixed precision training
- Faster optimization pipeline

---

## Evaluation

- Physical-world evaluation
- Patch robustness benchmarking
- Cross-dataset evaluation
- Transferability across detectors
- Video-based evaluation

---

## Detector Support

Future detector support may include:

- YOLOv9
- YOLOv10
- RT-DETR
- DETR
- Faster R-CNN

---

# Tech Stack

The framework is built using:

- Python
- PyTorch
- TorchVision
- Ultralytics YOLOv8
- OpenCV
- NumPy
- PyYAML

---

# Citation

If you find this repository useful in your research, please consider citing it.

```bibtex
@misc{shetty2026robustpatch,
  title={Robust Adversarial Patch Attack Framework for YOLOv8},
  author={Rishab Shetty},
  year={2026},
  note={Research Framework},
  url={https://github.com/RishabShetty16/robust-adversarial-patch-yolov8}
}
```

---

# License

This project is released under the **MIT License**.

Feel free to use, modify, and extend the framework for academic and research purposes.

---

# Author

## Rishab Shetty

**B.Tech — Computer Science & Engineering (Artificial Intelligence & Machine Learning)**

PES University, Bengaluru

### Research Interests

- Adversarial Machine Learning
- Computer Vision
- Deep Learning
- Artificial Intelligence
- AI Security
- Robust Machine Learning

---

# Acknowledgements

This project builds upon several outstanding open-source projects and research efforts.

Special thanks to:

- Ultralytics for the YOLOv8 framework
- PyTorch contributors
- The COCO dataset creators
- The open-source computer vision research community

---

# Project Status

**Current Stage:** Research Framework (Commit 31)

The repository now implements a complete end-to-end adversarial patch optimization framework featuring adaptive placement, configurable composite loss functions, confidence-weighted suppression, expectation over transformation, automatic checkpointing, comprehensive evaluation, and research-oriented experimentation.

The modular architecture is designed to facilitate future work in adversarial robustness, physical-world attacks, and transferable adversarial patch research.