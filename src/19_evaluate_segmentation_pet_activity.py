import numpy as np
from pathlib import Path
from scipy.ndimage import binary_dilation, binary_erosion

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "results" / "pet_registered_to_mri.npy"
mask_path = PROJECT_DIR / "results" / "tumor_mask_sam_3d.npy"

results_dir = PROJECT_DIR / "results"

pet = np.load(pet_path).astype(np.float32)
mask = np.load(mask_path).astype(bool)

print("=" * 60)
print("Segmentation assessment using registered PET activity")
print("=" * 60)

print("PET shape:", pet.shape)
print("Mask shape:", mask.shape)
print("Mask voxels:", mask.sum())

if pet.shape != mask.shape:
    raise ValueError("PET and mask must have the same shape.")

patient_region = pet > np.percentile(pet[pet > 0], 5)

# Original mask assessment
inside_values = pet[mask]

# PET-active rim around the MRI/SAM mask
outer_rim = binary_dilation(mask, iterations=3) & (~mask)
inner_rim = mask & (~binary_erosion(mask, iterations=3))
tumor_rim = (outer_rim | inner_rim) & patient_region

# Local background farther around tumor, excluding mask and rim
local_region = binary_dilation(mask, iterations=12) & patient_region
local_background = local_region & (~mask) & (~tumor_rim)

rim_values = pet[tumor_rim]
background_values = pet[local_background]

inside_mean = inside_values.mean()
inside_median = np.median(inside_values)
inside_max = inside_values.max()

rim_mean = rim_values.mean()
rim_median = np.median(rim_values)
rim_max = rim_values.max()

background_mean = background_values.mean()
background_median = np.median(background_values)
background_max = background_values.max()

inside_background_ratio = inside_mean / (background_mean + 1e-8)
rim_background_ratio = rim_mean / (background_mean + 1e-8)
rim_background_median_ratio = rim_median / (background_median + 1e-8)

print("\nPET activity inside filled SAM mask:")
print("Mean:", inside_mean)
print("Median:", inside_median)
print("Max:", inside_max)

print("\nPET activity on tumor rim/border:")
print("Rim voxels:", tumor_rim.sum())
print("Mean:", rim_mean)
print("Median:", rim_median)
print("Max:", rim_max)

print("\nPET activity in local background:")
print("Background voxels:", local_background.sum())
print("Mean:", background_mean)
print("Median:", background_median)
print("Max:", background_max)

print("\nPET activity ratios:")
print("Filled mask / background mean ratio:", inside_background_ratio)
print("Tumor rim / background mean ratio:", rim_background_ratio)
print("Tumor rim / background median ratio:", rim_background_median_ratio)

output_path = results_dir / "segmentation_pet_activity_assessment.txt"

with open(output_path, "w") as f:
    f.write("Segmentation assessment using registered PET activity\n")
    f.write("====================================================\n")
    f.write(f"Mask voxels: {int(mask.sum())}\n\n")

    f.write("PET activity inside filled SAM mask:\n")
    f.write(f"Mean: {inside_mean:.4f}\n")
    f.write(f"Median: {inside_median:.4f}\n")
    f.write(f"Max: {inside_max:.4f}\n")
    f.write(f"Filled mask/background mean ratio: {inside_background_ratio:.4f}\n\n")

    f.write("PET activity on tumor rim/border:\n")
    f.write(f"Rim voxels: {int(tumor_rim.sum())}\n")
    f.write(f"Mean: {rim_mean:.4f}\n")
    f.write(f"Median: {rim_median:.4f}\n")
    f.write(f"Max: {rim_max:.4f}\n")
    f.write(f"Rim/background mean ratio: {rim_background_ratio:.4f}\n")
    f.write(f"Rim/background median ratio: {rim_background_median_ratio:.4f}\n\n")

    f.write("PET activity in local background:\n")
    f.write(f"Background voxels: {int(local_background.sum())}\n")
    f.write(f"Mean: {background_mean:.4f}\n")
    f.write(f"Median: {background_median:.4f}\n")
    f.write(f"Max: {background_max:.4f}\n")

print("\nSaved assessment:")
print(output_path)

print("\nStep completed successfully.")