import torch

from attack.config import load_config
from attack.detector import YOLODetector

cfg = load_config("attack/configs/default.yaml")

detector = YOLODetector(cfg)

image = torch.rand(1, 3, 640, 640)

outputs = detector.forward(image)

print(type(outputs))

print()

print("Length:", len(outputs))

print()

for i, out in enumerate(outputs):

    print(f"Output {i}")

    print(type(out))

    if hasattr(out, "shape"):
        print(out.shape)

    elif isinstance(out, (list, tuple)):
        print("Length:", len(out))

        for j, x in enumerate(out):

            if hasattr(x, "shape"):
                print(f"   {j} -> {tuple(x.shape)}")

    print()