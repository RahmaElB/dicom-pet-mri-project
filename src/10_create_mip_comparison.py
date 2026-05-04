import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
pet_path = PROJECT_DIR / "results" / "pet_registered_to_mri.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)
pet = np.load(pet_path).astype(np.float32)

print("=" * 60)
print("Loaded registered images")
print("=" * 60)
print("MRI shape:", mri.shape)
print("Registered PET shape:", pet.shape)

if mri.shape != pet.shape:
    raise ValueError("MRI and registered PET must have the same shape.")

# Normalize images for visualization
mri_norm = (mri - np.percentile(mri, 5)) / (
    np.percentile(mri, 95) - np.percentile(mri, 5) + 1e-8
)
mri_norm = np.clip(mri_norm, 0, 1)

pet_norm = (pet - np.percentile(pet, 1)) / (
    np.percentile(pet, 99.5) - np.percentile(pet, 1) + 1e-8
)
pet_norm = np.clip(pet_norm, 0, 1)

# MRI: mean projection gives better anatomical visibility
# PET: maximum projection shows high-uptake regions
mri_mip = np.mean(mri_norm, axis=0)
pet_mip = np.max(pet_norm, axis=0)

print("\nMIP shapes:")
print("MRI MIP:", mri_mip.shape)
print("PET MIP:", pet_mip.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mri_mip, cmap="gray")
axes[0].set_title("MRI Reference MIP")
axes[0].axis("off")

axes[1].imshow(pet_mip, cmap="hot")
axes[1].set_title("Registered PET MIP")
axes[1].axis("off")

axes[2].imshow(mri_mip, cmap="gray")
axes[2].imshow(pet_mip, cmap="hot", alpha=0.45)
axes[2].set_title("MRI + PET Alpha Fusion MIP")
axes[2].axis("off")

plt.tight_layout()

figure_path = figures_dir / "mip_mri_pet_fusion_static.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved MIP comparison figure:")
print(figure_path)

print("\nStep 11 completed successfully.")