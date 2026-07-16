from attack.config import load_config, get_device

cfg = load_config("attack/configs/default.yaml")

print("=" * 60)
print("Experiment Configuration")
print("=" * 60)

print("Experiment :", cfg["experiment"]["name"])
print("Device     :", cfg["device"])
print("Model      :", cfg["model"]["weights"])
print("Patch Size :", cfg["patch"]["size"])
print("Optimizer  :", cfg["optimizer"]["type"])
print("LR         :", cfg["optimizer"]["lr"])

# Print optional fields only if they exist
if "training" in cfg:
    if "iterations" in cfg["training"]:
        print("Iterations :", cfg["training"]["iterations"])

if "attack" in cfg:
    if "type" in cfg["attack"]:
        print("Attack     :", cfg["attack"]["type"])

print("=" * 60)

print(get_device(cfg))