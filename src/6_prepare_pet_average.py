import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

pet_4d = np.load(pet_path)

print("=" * 60)
print("Loaded PET 4D")
print("=" * 60)
print("PET 4D shape:", pet_4d.shape)
print("Format: time, slice, row, column")

# Average across time axis
pet_average = pet_4d.mean(axis=0).astype(np.float32)

print("\n" + "=" * 60)
print("PET average volume")
print("=" * 60)
print("PET average shape:", pet_average.shape)
print("PET average dtype:", pet_average.dtype)
print("PET average min:", pet_average.min())
print("PET average max:", pet_average.max())

# Save PET average volume
output_path = results_dir / "pet_average.npy"
np.save(output_path, pet_average)

print("\nSaved PET average file:")
print(output_path)

# Visualize middle slice
middle_slice = pet_average.shape[0] // 2
pet_middle = pet_average[middle_slice]

plt.figure(figsize=(6, 6))
plt.imshow(pet_middle, cmap="gray")
plt.title(f"PET Average Volume - Middle Slice {middle_slice}")
plt.axis("off")
plt.colorbar()
plt.savefig(figures_dir / "pet_average_volume_middle_slice.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figures_dir / "pet_average_volume_middle_slice.png")

print("\nStep 7 completed successfully.")