from attack.config import load_config
from attack.detector import YOLODetector
from attack.parser import DetectionParser

cfg = load_config("attack/configs/default.yaml")

detector = YOLODetector(cfg)

results = detector("data/coco/images/person.jpg")

parser = DetectionParser()

detections = parser.parse(results)

print(parser)

print()
print("Detections Found :", len(detections))
print()

for detection in detections:
    print(detection)