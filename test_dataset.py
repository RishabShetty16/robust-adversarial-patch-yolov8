from attack.config import load_config
from attack.dataset import COCODataset

cfg = load_config("attack/configs/default.yaml")

dataset = COCODataset(cfg)

print(dataset)

print()

print("Dataset Length :", len(dataset))

if len(dataset):

    image = dataset[0]

    print("Shape :", image.shape)