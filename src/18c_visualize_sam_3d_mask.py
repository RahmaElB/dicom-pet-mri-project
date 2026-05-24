import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
mask_path = PROJECT_DIR / "results" / "tumor_mask_sam_3d.npy"

figures_dir = PROJECT_DIR / "results" / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)
mask = np.load(mask_path).astype(bool)

print("=" * 60)
print("3D SAM mask visualization")
print("=" * 60)

print("MRI shape:", mri.shape)
print("Mask shape:", mask.shape)
print("Mask voxels:", mask.sum())

if mri.shape != mask.shape:
    raise ValueError("MRI and mask must have the same shape.")

# Normalize MRI
mri_norm = (mri - np.percentile(mri, 1)) / (
    np.percentile(mri, 99) - np.percentile(mri, 1) + 1e-8
)
mri_norm = np.clip(mri_norm, 0, 1)

# Projections
mri_axial_mip = np.mean(mri_norm, axis=0)
mask_axial_mip = np.max(mask, axis=0)

mri_coronal_mip = np.mean(mri_norm, axis=1)
mask_coronal_mip = np.max(mask, axis=1)

mri_sagittal_mip = np.mean(mri_norm, axis=2)
mask_sagittal_mip = np.max(mask, axis=2)


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mri_axial_mip, cmap="gray")
axes[0].imshow(mask_axial_mip, cmap="Reds", alpha=0.45)
axes[0].set_title("Axial Mean Projection + 3D Mask")

axes[1].imshow(mri_coronal_mip, cmap="gray", aspect="auto")
axes[1].imshow(mask_coronal_mip, cmap="Reds", alpha=0.45, aspect="auto")
axes[1].set_title("Coronal Mean Projection + 3D Mask")

axes[2].imshow(mri_sagittal_mip, cmap="gray", aspect="auto")
axes[2].imshow(mask_sagittal_mip, cmap="Reds", alpha=0.45, aspect="auto")
axes[2].set_title("Sagittal Mean Projection + 3D Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

figure_path = figures_dir / "sam_3d_mask_mip_views.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

print("\nStep completed successfully.")