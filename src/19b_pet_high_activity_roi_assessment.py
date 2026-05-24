import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_dilation

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "results" / "pet_registered_to_mri.npy"
mri_path = PROJECT_DIR / "results" / "mri.npy"
sam_mask_path = PROJECT_DIR / "results" / "tumor_mask_sam_3d.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

pet = np.load(pet_path).astype(np.float32)
mri = np.load(mri_path).astype(np.float32)
sam_mask = np.load(sam_mask_path).astype(bool)

print("=" * 60)
print("PET high-activity ROI assessment")
print("=" * 60)

print("PET shape:", pet.shape)
print("MRI shape:", mri.shape)
print("SAM mask shape:", sam_mask.shape)

if pet.shape != mri.shape or pet.shape != sam_mask.shape:
    raise ValueError("PET, MRI, and SAM mask must have the same shape.")

# Use the same lesion region used for SAM
slice_start = 72
slice_end = 80

x_min = 145
x_max = 195
y_min = 155
y_max = 215

roi_mask = np.zeros_like(sam_mask, dtype=bool)
roi_mask[slice_start:slice_end, y_min:y_max, x_min:x_max] = True

pet_roi_values = pet[roi_mask]

# Threshold high PET uptake inside the lesion ROI
threshold = np.percentile(pet_roi_values, 85)
pet_high_mask = roi_mask & (pet >= threshold)

# Local background around ROI, excluding ROI
local_region = binary_dilation(roi_mask, iterations=15)
patient_region = pet > np.percentile(pet[pet > 0], 5)
local_background = local_region & (~roi_mask) & patient_region

pet_high_values = pet[pet_high_mask]
background_values = pet[local_background]
sam_values = pet[sam_mask]

pet_high_mean = pet_high_values.mean()
pet_high_median = np.median(pet_high_values)
pet_high_max = pet_high_values.max()

background_mean = background_values.mean()
background_median = np.median(background_values)

sam_mean = sam_values.mean()
sam_median = np.median(sam_values)

high_to_background_mean_ratio = pet_high_mean / (background_mean + 1e-8)
high_to_background_median_ratio = pet_high_median / (background_median + 1e-8)

print("\nROI information:")
print("Slices:", slice_start, "to", slice_end - 1)
print("Box:", x_min, y_min, x_max, y_max)
print("ROI voxels:", roi_mask.sum())

print("\nPET high-activity threshold inside ROI:")
print("85th percentile threshold:", threshold)
print("PET-high voxels:", pet_high_mask.sum())

print("\nPET activity in PET-high lesion region:")
print("Mean:", pet_high_mean)
print("Median:", pet_high_median)
print("Max:", pet_high_max)

print("\nPET activity in local background:")
print("Background voxels:", local_background.sum())
print("Mean:", background_mean)
print("Median:", background_median)

print("\nPET-high lesion/background ratios:")
print("Mean ratio:", high_to_background_mean_ratio)
print("Median ratio:", high_to_background_median_ratio)

print("\nPET activity inside SAM MRI mask:")
print("Mean:", sam_mean)
print("Median:", sam_median)

# Save PET high mask
pet_high_mask_path = results_dir / "pet_high_activity_mask_roi.npy"
np.save(pet_high_mask_path, pet_high_mask)

# Save text summary
summary_path = results_dir / "pet_high_activity_roi_assessment.txt"
with open(summary_path, "w") as f:
    f.write("PET high-activity ROI assessment\n")
    f.write("================================\n")
    f.write(f"Slices: {slice_start} to {slice_end - 1}\n")
    f.write(f"Box: [{x_min}, {y_min}, {x_max}, {y_max}]\n")
    f.write(f"ROI voxels: {int(roi_mask.sum())}\n\n")

    f.write(f"PET 85th percentile threshold inside ROI: {threshold:.4f}\n")
    f.write(f"PET-high voxels: {int(pet_high_mask.sum())}\n\n")

    f.write("PET activity in PET-high lesion region:\n")
    f.write(f"Mean: {pet_high_mean:.4f}\n")
    f.write(f"Median: {pet_high_median:.4f}\n")
    f.write(f"Max: {pet_high_max:.4f}\n\n")

    f.write("PET activity in local background:\n")
    f.write(f"Background voxels: {int(local_background.sum())}\n")
    f.write(f"Mean: {background_mean:.4f}\n")
    f.write(f"Median: {background_median:.4f}\n\n")

    f.write("PET-high lesion/background ratios:\n")
    f.write(f"Mean ratio: {high_to_background_mean_ratio:.4f}\n")
    f.write(f"Median ratio: {high_to_background_median_ratio:.4f}\n\n")

    f.write("PET activity inside SAM MRI mask:\n")
    f.write(f"Mean: {sam_mean:.4f}\n")
    f.write(f"Median: {sam_median:.4f}\n")

print("\nSaved PET-high mask:")
print(pet_high_mask_path)

print("\nSaved summary:")
print(summary_path)

# Visualization on central slice
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

fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(mri_norm, cmap="gray")
axes[0].set_title("MRI Slice 78")

axes[1].imshow(pet_norm, cmap="hot")
axes[1].set_title("Registered PET Slice 78")

axes[2].imshow(mri_norm, cmap="gray")
axes[2].imshow(sam_slice, cmap="Reds", alpha=0.45)
axes[2].set_title("MRI + SAM Mask")

axes[3].imshow(mri_norm, cmap="gray")
axes[3].imshow(pet_high_slice, cmap="hot", alpha=0.55)
axes[3].set_title("MRI + PET-High ROI Mask")

for ax in axes:
    ax.axis("off")

plt.tight_layout()

figure_path = figures_dir / "pet_high_activity_roi_assessment_slice78.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

print("\nStep completed successfully.")