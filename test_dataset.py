from attack.config import load_config
from attack.dataset import COCODataset

cfg = load_config("attack/configs/default.yaml")

dataset = COCODataset(cfg)

print(dataset)

image = dataset[0]

print()
print("Dataset Length :", len(dataset))
print("Shape :", image.shape)