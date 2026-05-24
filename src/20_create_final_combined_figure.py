import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri = np.load(PROJECT_DIR / "results" / "mri.npy").astype(np.float32)
pet = np.load(PROJECT_DIR / "results" / "pet_registered_to_mri.npy").astype(np.float32)
sam_mask = np.load(PROJECT_DIR / "results" / "tumor_mask_sam_3d.npy").astype(bool)
pet_high_mask = np.load(PROJECT_DIR / "results" / "pet_high_activity_mask_roi.npy").astype(bool)

figures_dir = PROJECT_DIR / "results" / "figures"
figures_dir.mkdir(exist_ok=True)

slice_index = 78

mri_slice = mri[slice_index]
pet_slice = pet[slice_index]
sam_slice = sam_mask[slice_index]
pet_high_slice = pet_high_mask[slice_index]

mri_norm = (mri_slice - np.percentile(mri_slice, 1)) / (
    np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1) + 1e-8
)
mri_norm = np.clip(mri_norm, 0, 1)

pet_norm = (pet_slice - np.percentile(pet_slice, 1)) / (
    np.percentile(pet_slice, 99.5) - np.percentile(pet_slice, 1) + 1e-8
)
pet_norm = np.clip(pet_norm, 0, 1)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(mri_norm, cmap="gray")
axes[0, 0].set_title("MRI Reference")

axes[0, 1].imshow(pet_norm, cmap="hot")
axes[0, 1].set_title("Registered PET")

axes[0, 2].imshow(mri_norm, cmap="gray")
axes[0, 2].imshow(pet_norm, cmap="hot", alpha=0.4)
axes[0, 2].set_title("MRI + Registered PET")

axes[1, 0].imshow(mri_norm, cmap="gray")
axes[1, 0].imshow(sam_slice, cmap="Reds", alpha=0.45)
axes[1, 0].set_title("MRI + 3D SAM Mask")

axes[1, 1].imshow(mri_norm, cmap="gray")
axes[1, 1].imshow(pet_high_slice, cmap="hot", alpha=0.6)
axes[1, 1].set_title("MRI + PET-High ROI")

axes[1, 2].imshow(mri_norm, cmap="gray")
axes[1, 2].imshow(sam_slice, cmap="Reds", alpha=0.35)
axes[1, 2].imshow(pet_high_slice, cmap="hot", alpha=0.55)
axes[1, 2].set_title("SAM Mask + PET-High Region")

for ax in axes.ravel():
    ax.axis("off")

plt.suptitle("Final Multimodal PET-MRI Registration and Tumor Segmentation Results", fontsize=16)
plt.tight_layout()

figure_path = figures_dir / "final_combined_pet_mri_segmentation_results.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved final combined figure:")
print(figure_path)