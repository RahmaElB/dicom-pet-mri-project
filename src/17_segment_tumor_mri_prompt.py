import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from scipy.ndimage import binary_closing, binary_fill_holes, label

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)

print("=" * 60)
print("MRI seeded tumor segmentation")
print("=" * 60)
print("MRI shape:", mri.shape)

# Best MRI slice
slice_index = 78

# Tight box around lesion
x_min = 145
x_max = 195
y_min = 155   # shifted DOWN
y_max = 215

mri_slice = mri[slice_index]
mri_roi = mri_slice[y_min:y_max, x_min:x_max]

print("MRI ROI shape:", mri_roi.shape)

# Seed point inside dark lesion, in full image coordinates
seed_x = 170
seed_y = 185

# Convert seed to ROI coordinates
seed_rx = seed_x - x_min
seed_ry = seed_y - y_min

print("Seed in ROI:", seed_rx, seed_ry)

# Dark lesion threshold
threshold = np.percentile(mri_roi, 45)
candidate_mask = mri_roi < threshold

# Connected components
labeled_mask, num_features = label(candidate_mask)

seed_label = labeled_mask[seed_ry, seed_rx]

if seed_label == 0:
    raise RuntimeError(
        "Seed is not inside segmented dark region. Move seed_x/seed_y deeper into lesion."
    )

mask_roi = labeled_mask == seed_label

# Clean
mask_roi = binary_closing(mask_roi, iterations=2)
mask_roi = binary_fill_holes(mask_roi)

mask_slice = np.zeros_like(mri_slice, dtype=bool)
mask_slice[y_min:y_max, x_min:x_max] = mask_roi

np.save(results_dir / "tumor_mask_mri_slice78.npy", mask_slice)

print("Mask pixels:", mask_roi.sum())

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

for ax in axes:
    ax.axis("off")

axes[0].imshow(mri_slice, cmap="gray")
axes[0].set_title(f"MRI Slice {slice_index}")
rect = patches.Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
axes[0].add_patch(rect)
axes[0].plot(seed_x, seed_y, "bo", markersize=5)

axes[1].imshow(mri_roi, cmap="gray")
axes[1].set_title("MRI ROI")
axes[1].plot(seed_rx, seed_ry, "bo", markersize=5)

axes[2].imshow(mask_roi, cmap="gray")
axes[2].set_title("Seeded Tumor Mask")

axes[3].imshow(mri_roi, cmap="gray")
axes[3].imshow(mask_roi, cmap="Reds", alpha=0.45)
axes[3].set_title("Mask Overlay on MRI ROI")

plt.tight_layout()

figure_path = figures_dir / "mri_seeded_tumor_segmentation.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)
print("\nStep completed.")