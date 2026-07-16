from attack.patch import AdversarialPatch

patch = AdversarialPatch(size=160)

print(patch)

tensor = patch()

print()
print("Statistics")
print("shape:", tuple(tensor.shape))
print("mean:", tensor.mean().item())
print("std:", tensor.std().item())
print("min:", tensor.min().item())
print("max:", tensor.max().item())

patch.save("outputs/patches/test_patch.png")

print()
print("Patch saved successfully.")