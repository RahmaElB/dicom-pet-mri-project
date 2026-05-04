import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import binary_opening, binary_closing

PROJECT_DIR = Path(__file__).resolve().parents[1]

roi_path = PROJECT_DIR / "results" / "tumor_roi_pet_slice23.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

# Load ROI
tumor_roi = np.load(roi_path).astype(np.float32)

print("=" * 60)
print("Loaded tumor ROI")
print("=" * 60)
print("ROI shape:", tumor_roi.shape)
print("ROI min:", tumor_roi.min())
print("ROI max:", tumor_roi.max())
print("ROI mean:", tumor_roi.mean())


threshold = np.percentile(tumor_roi, 85)

print("\nThreshold value:", threshold)

tumor_mask = tumor_roi > threshold   


# CLEAN MASK (remove noise)

tumor_mask = binary_opening(tumor_mask, iterations=2)
tumor_mask = binary_closing(tumor_mask, iterations=2)

from scipy.ndimage import label

labeled_mask, num_features = label(tumor_mask)

print("Connected components found:", num_features)

# Compute center of ROI
center_y, center_x = tumor_mask.shape[0] // 2, tumor_mask.shape[1] // 2

# Find which label is closest to center
best_label = None
best_intensity = -np.inf

for label_id in range(1, num_features + 1):
    coords = np.argwhere(labeled_mask == label_id)
    if len(coords) == 0:
        continue

    # mean intensity of this region
    intensities = tumor_roi[coords[:, 0], coords[:, 1]]
    mean_intensity = intensities.mean()

    if mean_intensity > best_intensity:
        best_intensity = mean_intensity
        best_label = label_id

# Keep only that component
tumor_mask = labeled_mask == best_label

print("Mask shape:", tumor_mask.shape)
print("Mask pixels:", tumor_mask.sum())

# Save mask
mask_path = results_dir / "tumor_mask_pet_slice23.npy"
np.save(mask_path, tumor_mask)

print("\nSaved tumor mask:")
print(mask_path)


# Visualization

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(tumor_roi, cmap="gray")
axes[0].set_title("Tumor ROI")
axes[0].axis("off")

axes[1].imshow(tumor_mask, cmap="gray")
axes[1].set_title("Cleaned Tumor Mask")
axes[1].axis("off")

axes[2].imshow(tumor_roi, cmap="gray")
axes[2].imshow(tumor_mask, cmap="Reds", alpha=0.4)
axes[2].set_title("Mask Overlay on ROI")
axes[2].axis("off")

plt.tight_layout()

figure_path = figures_dir / "tumor_threshold_segmentation_pet_slice23.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

print("\nStep 16 completed successfully.")