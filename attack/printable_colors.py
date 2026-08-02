"""
printable_colors.py

Printable RGB color palette used for the
Non-Printability Score (NPS).

Author:
    Rishab Shetty
"""

import torch

PRINTABLE_COLORS = torch.tensor(
    [
        [0.0, 0.0, 0.0],      # Black
        [1.0, 1.0, 1.0],      # White
        [1.0, 0.0, 0.0],      # Red
        [0.0, 1.0, 0.0],      # Green
        [0.0, 0.0, 1.0],      # Blue
        [1.0, 1.0, 0.0],      # Yellow
        [1.0, 0.0, 1.0],      # Magenta
        [0.0, 1.0, 1.0],      # Cyan
        [0.5, 0.5, 0.5],      # Gray
        [0.75, 0.75, 0.75],   # Light Gray
        [0.25, 0.25, 0.25],   # Dark Gray
        [0.5, 0.0, 0.0],      # Dark Red
        [0.0, 0.5, 0.0],      # Dark Green
        [0.0, 0.0, 0.5],      # Dark Blue
        [1.0, 0.5, 0.0],      # Orange
        [0.6, 0.3, 0.1],      # Brown
    ],
    dtype=torch.float32,
)