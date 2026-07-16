import torch

from attack.losses import (
    confidence_loss,
    objectness_loss,
    total_variation_loss,
    non_printability_score,
)

confidences = torch.tensor(
    [0.95, 0.80, 0.63],
    requires_grad=True,
)

objectness = torch.tensor(
    [0.91, 0.84],
    requires_grad=True,
)

patch = torch.rand(
    3,
    160,
    160,
    requires_grad=True,
)

print("Confidence :", confidence_loss(confidences))
print("Objectness :", objectness_loss(objectness))
print("TV :", total_variation_loss(patch))
print("NPS :", non_printability_score(patch))