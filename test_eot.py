from attack.config import load_config
from attack.eot import EOT

cfg = load_config("attack/configs/default.yaml")

eot = EOT(cfg)

print(eot)