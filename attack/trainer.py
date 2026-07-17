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

from attack.losses import baseline_loss
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

        # -------------------------------------------------
        # Attack Target
        # -------------------------------------------------

        self.attack_target = AttackTarget(cfg)

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    def train(self):

        print("=" * 60)
        print("Patch Trainer")
        print("=" * 60)

        # -------------------------------------------------
        # Load Image
        # -------------------------------------------------

        image = self.dataset[0]

        print("Image Shape :", image.shape)

        # -------------------------------------------------
        # Generate Adversarial Patch
        # -------------------------------------------------

        patch = self.patch()

        print("Patch Shape :", patch.shape)

        # -------------------------------------------------
        # Apply Patch
        # -------------------------------------------------

        patched = self.patch_applier.apply(
            image=image,
            patch=patch,
            x=100,
            y=100,
        )

        print("Patched Image :", patched.shape)

        # -------------------------------------------------
        # Add Batch Dimension
        # -------------------------------------------------

        patched = patched.unsqueeze(0)

        print("Batch Shape :", patched.shape)

        print("=" * 60)

        # -------------------------------------------------
        # YOLO Forward Pass
        # -------------------------------------------------

        outputs = self.detector.forward(patched)

        print()
        print("Forward Pass Successful")

        # -------------------------------------------------
        # Extract Attack Targets
        # -------------------------------------------------

        targets = self.attack_target.extract(outputs)

        predictions = targets["predictions"]

        print("Prediction Shape :", predictions.shape)

        # -------------------------------------------------
        # Baseline Optimization Loss
        # -------------------------------------------------

        loss = baseline_loss(predictions)

        print()
        print("Loss :", loss.item())

        # -------------------------------------------------
        # Backpropagation
        # -------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        print()
        print("Gradient Exists :", self.patch.patch.grad is not None)

        if self.patch.patch.grad is not None:
            print("Gradient Shape :", self.patch.patch.grad.shape)

        # -------------------------------------------------
        # Optimizer Update
        # -------------------------------------------------

        self.optimizer.step()

        print()
        print("Optimizer Step Completed")
        print("=" * 60)