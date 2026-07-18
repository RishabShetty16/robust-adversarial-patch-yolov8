"""
detector.py

YOLOv8 Detector Wrapper

Responsibilities
----------------
- Load YOLO model
- Run inference
- Return raw YOLO predictions

Author:
    Rishab Shetty
"""

from pathlib import Path

from ultralytics import YOLO
import torch


class YOLODetector:
    """
    Wrapper around Ultralytics YOLO.
    """

    def __init__(self, cfg):

        self.cfg = cfg

        model_cfg = cfg["model"]

        self.weights = model_cfg["weights"]
        self.confidence = model_cfg["confidence"]
        self.iou = model_cfg["iou"]

        self.model = None

        self.load_model()

    # -----------------------------------------------------
    # Load model
    # -----------------------------------------------------

    def load_model(self):

        print("=" * 60)
        print("Loading YOLO Model")
        print("=" * 60)

        self.model = YOLO(self.weights)

        print(f"Weights : {self.weights}")

        print("=" * 60)

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    @torch.no_grad()
    def predict(self, image):

        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            verbose=False,
        )

        return results

    # -----------------------------------------------------
    # Callable
    # -----------------------------------------------------

    def __call__(self, image):

        return self.predict(image)

    # -----------------------------------------------------

    def __repr__(self):

        return (
            f"YOLODetector("
            f"weights='{self.weights}')"
        )
    
    # -----------------------------------------------------
    # Forward (Training)
    # -----------------------------------------------------

    def forward(self, image):
        """
        Forward pass for training.
        """

        outputs = self.model.model(image)

        print("\n========== DETECTOR DEBUG ==========")
        print("Type:", type(outputs))

        if isinstance(outputs, (list, tuple)):
            print("Length:", len(outputs))

            for i, out in enumerate(outputs):
                if hasattr(out, "shape"):
                    print(f"Output[{i}] Shape:", out.shape)
                else:
                    print(f"Output[{i}] Type:", type(out))
        else:
            if hasattr(outputs, "shape"):
                print("Shape:", outputs.shape)

        print("====================================\n")

        return outputs


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    from attack.config import load_config

    cfg = load_config("attack/configs/default.yaml")

    detector = YOLODetector(cfg)

    print(detector)

    sample = "data/sample/person.jpg"

    if Path(sample).exists():

        results = detector(sample)

        print()

        print("Images Processed :", len(results))

        print("Inference Successful")

    else:

        print()

        print("Sample image not found:")
        print(sample)