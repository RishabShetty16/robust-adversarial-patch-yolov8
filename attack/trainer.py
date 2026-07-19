"""
trainer.py

Training engine for adversarial patch optimization.

Responsibilities
----------------
- Dataset iteration
- Patch application
- YOLO forward pass
- Attack target extraction
- Loss computation
- Backpropagation
- Optimizer update
- Logging
- Saving checkpoints

Author:
    Rishab Shetty
"""

import os
import csv
import torch

from attack.eot import EOT
from torch.utils.data import DataLoader

from attack.losses import person_suppression_loss
from attack.attack_target import AttackTarget


class PatchTrainer:
    """
    Trainer for adversarial patch optimization.
    """

    def __init__(
        self,
        cfg,
        detector,
        dataset,
        patch,
        optimizer,
        patch_applier,
    ):
        self.cfg = cfg
        self.detector = detector
        self.dataset = dataset
        self.patch = patch
        self.optimizer = optimizer
        self.patch_applier = patch_applier
        self.eot = EOT(cfg)

        self.attack_target = AttackTarget(cfg)

        # -------------------------------------------------
        # DataLoader
        # -------------------------------------------------

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["training"]["num_workers"],
        )

        # -------------------------------------------------
        # Training History
        # -------------------------------------------------

        self.loss_history = []

    # -------------------------------------------------
    # Random Patch Placement
    # -------------------------------------------------

    def random_patch_position(self, image, patch):
        """
        Generate a random valid patch location.
        """

        _, _, image_h, image_w = image.shape
        _, patch_h, patch_w = patch.shape

        x = torch.randint(
            low=0,
            high=image_w - patch_w + 1,
            size=(1,),
        ).item()

        y = torch.randint(
            low=0,
            high=image_h - patch_h + 1,
            size=(1,),
        ).item()

        return x, y

    # -------------------------------------------------
    # Main Training Loop
    # -------------------------------------------------

    def train(self):
        """
        Entry point for training.
        """

        print("=" * 60)
        print("Patch Trainer")
        print("=" * 60)

        epochs = self.cfg["training"]["epochs"]

        for epoch in range(epochs):

            print()
            print("=" * 60)
            print(f"Epoch {epoch + 1}/{epochs}")
            print("=" * 60)

            average_loss = self.train_epoch()

            self.loss_history.append(average_loss)

            self.log_metrics(epoch, average_loss)

            self.save_checkpoint(epoch)

        self.save_loss_history()

    # -------------------------------------------------
    # Train One Epoch
    # -------------------------------------------------

    def train_epoch(self):
        """
        Executes one training epoch.
        """

        epoch_loss = 0.0

        for image in self.dataloader:

            loss = self.train_step(image)

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(self.dataloader)

        return average_loss

    # -------------------------------------------------
    # Train One Step
    # -------------------------------------------------

    def train_step(self, image):
        """
        Executes one optimization step.
        """

        print("Batch Shape :", image.shape)

        patch = self.patch()
        patch = self.eot(patch)

        print("Patch Shape :", patch.shape)

        # -------------------------------------------------
        # Random Patch Position
        # -------------------------------------------------

        x, y = self.random_patch_position(
            image,
            patch,
        )

        print(f"Patch Position : ({x}, {y})")

        patched = self.patch_applier.apply(
            image=image,
            patch=patch,
            x=x,
            y=y,
        )

        print("Patched Batch Shape :", patched.shape)

        print("=" * 60)

        outputs = self.detector.forward(patched)

        print()
        print("Forward Pass Successful")

        targets = self.attack_target.extract(outputs)

        target_scores = targets["target_scores"]

        print("Target Scores Shape :", target_scores.shape)

        assert target_scores.ndim == 2, (
            f"Expected target_scores shape (B, N), "
            f"got {target_scores.shape}"
        )

        loss = person_suppression_loss(target_scores)

        print()
        print("Person Suppression Loss :", loss.item())

        print()
        print(
            f"Mean Target Confidence : {target_scores.mean().item():.6f}"
        )

        print(
            f"Max Target Confidence  : {target_scores.max().item():.6f}"
        )

        print(
            f"Min Target Confidence  : {target_scores.min().item():.6f}"
        )

        self.optimizer.zero_grad()

        loss.backward()

        print()
        print("Gradient Exists :", self.patch.patch.grad is not None)

        if self.patch.patch.grad is not None:
            print("Gradient Shape :", self.patch.patch.grad.shape)

        self.optimizer.step()

        with torch.no_grad():
            self.patch.patch.clamp_(
                self.cfg["patch"]["clamp_min"],
                self.cfg["patch"]["clamp_max"],
            )

        print()
        print("Patch Mean :", self.patch().mean().item())
        print("Patch Min  :", self.patch().min().item())
        print("Patch Max  :", self.patch().max().item())

        print()
        print("Optimizer Step Completed")
        print("=" * 60)

        return loss

    # -------------------------------------------------
    # Logging
    # -------------------------------------------------

    def log_metrics(self, epoch, average_loss):
        """
        Logs training metrics.
        """

        patch = self.patch()

        print()
        print("-" * 60)
        print(f"Epoch        : {epoch + 1}")
        print(f"Average Loss : {average_loss:.6f}")
        print()
        print(f"Patch Mean   : {patch.mean().item():.6f}")
        print(f"Patch Std    : {patch.std().item():.6f}")
        print(f"Patch Min    : {patch.min().item():.6f}")
        print(f"Patch Max    : {patch.max().item():.6f}")

    # -------------------------------------------------
    # Save Checkpoint
    # -------------------------------------------------

    def save_checkpoint(self, epoch):
        """
        Save the adversarial patch checkpoint.
        """

        save_every = self.cfg["training"]["save_every"]

        if (epoch + 1) % save_every != 0:
            return

        checkpoint_dir = "outputs/checkpoints"

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"epoch_{epoch + 1:03d}.pt",
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "patch": self.patch.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

        print()
        print(f"Checkpoint Saved : {checkpoint_path}")

    # -------------------------------------------------
    # Save Loss History
    # -------------------------------------------------

    def save_loss_history(self):
        """
        Save training loss history to CSV.
        """

        log_dir = "outputs/logs"

        os.makedirs(log_dir, exist_ok=True)

        log_file = os.path.join(
            log_dir,
            "loss_history.csv",
        )

        with open(log_file, "w", newline="") as file:

            writer = csv.writer(file)

            writer.writerow(["Epoch", "Average Loss"])

            for i, loss in enumerate(self.loss_history):

                writer.writerow([i + 1, loss])

        print()
        print(f"Loss History Saved : {log_file}")