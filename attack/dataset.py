"""
dataset.py

COCO Dataset Loader

This module loads images from the COCO dataset and prepares them
for adversarial patch training.

Responsibilities
----------------
- Read images
- Resize images
- Convert to tensor
- Return image tensor

Author:
    Rishab Shetty
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class COCODataset(Dataset):
    """
    Dataset for loading COCO images.
    """

    def __init__(self, cfg):

        # Read configuration
        dataset_cfg = cfg["dataset"]

        self.root = Path(dataset_cfg["root"])
        self.image_dir = self.root / dataset_cfg["images"]

        self.image_size = dataset_cfg["image_size"]

        # Collect image paths
        self.image_paths = sorted(
            list(self.image_dir.glob("*.jpg")) +
            list(self.image_dir.glob("*.png")) +
            list(self.image_dir.glob("*.jpeg"))
        )

        max_images = dataset_cfg.get("max_images")

        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ])

        print("=" * 60)
        print("COCO Dataset Initialized")
        print("=" * 60)
        print(f"Image Directory : {self.image_dir}")
        print(f"Image Size      : {self.image_size}")
        print(f"Images Found    : {len(self.image_paths)}")
        print("=" * 60)

    # --------------------------------------------------
    # Number of images
    # --------------------------------------------------

    def __len__(self):

        return len(self.image_paths)

    # --------------------------------------------------
    # Read image
    # --------------------------------------------------

    def _load_image(self, image_path):

        image = Image.open(image_path).convert("RGB")

        return image

    # --------------------------------------------------
    # Get one sample
    # --------------------------------------------------

    def __getitem__(self, index):

        image_path = self.image_paths[index]

        image = self._load_image(image_path)

        image = self.transform(image)

        return image

    # --------------------------------------------------
    # Pretty Representation
    # --------------------------------------------------

    def __repr__(self):

        return (
            f"COCODataset("
            f"images={len(self)}, "
            f"image_size={self.image_size})"
        )


# ==========================================================
# Test
# ==========================================================
