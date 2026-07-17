import torch

from attack.config import load_config
from attack.detector import YOLODetector

cfg = load_config("attack/configs/default.yaml")

detector = YOLODetector(cfg)

image = torch.rand(1,3,640,640)

predictions = detector.forward(image)[0]

print("="*60)
print("Prediction Tensor")
print("="*60)

print(predictions.shape)

print()

print("First 10 channels")

for i in range(10):

    print(
        i,
        predictions[0,i,:5]
    )