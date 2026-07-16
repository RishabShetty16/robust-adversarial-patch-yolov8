from attack.config import load_config
from attack.detector import YOLODetector

cfg = load_config("attack/configs/default.yaml")

detector = YOLODetector(cfg)

print(detector)

results = detector("data/coco/images/person.jpg")

print()
print("Results Returned :", len(results))