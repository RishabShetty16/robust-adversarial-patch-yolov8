import torch

from attack.config import load_config
from attack.detector import YOLODetector

cfg = load_config("attack/configs/default.yaml")

detector = YOLODetector(cfg)

image = torch.rand(1, 3, 640, 640)

outputs = detector.forward(image)

pred = outputs[0]

print("Prediction Shape :", pred.shape)
print()

print("Minimum :", pred.min().item())
print("Maximum :", pred.max().item())
print("Mean    :", pred.mean().item())

print()
print("First Prediction Vector")
print(pred[0, :, 0])