import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
pet_path = PROJECT_DIR / "results" / "pet_average.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)
pet = np.load(pet_path).astype(np.float32)

print("=" * 60)
print("Loaded images")
print("=" * 60)
print("MRI shape:", mri.shape)
print("PET shape:", pet.shape)

mri_img = sitk.GetImageFromArray(mri)
pet_img = sitk.GetImageFromArray(pet)

mri_img.SetSpacing((1.0, 1.0, 1.0))
pet_img.SetSpacing((1.171875, 1.171875, 3.27))

print("\nMRI size:", mri_img.GetSize())
print("MRI spacing:", mri_img.GetSpacing())

print("\nPET size:", pet_img.GetSize())
print("PET spacing:", pet_img.GetSpacing())

identity_transform = sitk.Transform(3, sitk.sitkIdentity)

pet_resampled_img = sitk.Resample(
    pet_img,
    mri_img,
    identity_transform,
    sitk.sitkLinear,
    0.0,
    pet_img.GetPixelID()
)

pet_resampled = sitk.GetArrayFromImage(pet_resampled_img).astype(np.float32)

print("\n" + "=" * 60)
print("Resampled PET")
print("=" * 60)
print("Resampled PET shape:", pet_resampled.shape)
print("Resampled PET min:", pet_resampled.min())
print("Resampled PET max:", pet_resampled.max())

output_path = results_dir / "pet_average_resampled_to_mri_space.npy"
np.save(output_path, pet_resampled)

print("\nSaved resampled PET:")
print(output_path)

# Visualize MRI and resampled PET side by side
middle_slice = mri.shape[0] // 2

mri_slice = mri[middle_slice]
pet_slice = pet_resampled[middle_slice]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(mri_slice, cmap="gray")
axes[0].set_title("MRI Middle Slice")
axes[0].axis("off")

axes[1].imshow(pet_slice, cmap="gray")
axes[1].set_title("PET Resampled to MRI Space")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(figures_dir / "mri_pet_resampled_side_by_side.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figures_dir / "mri_pet_resampled_side_by_side.png")

print("\nStep 9 completed successfully.")