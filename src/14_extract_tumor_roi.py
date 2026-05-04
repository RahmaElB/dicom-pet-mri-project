import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_4d_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"
bbox_path = PROJECT_DIR / "results" / "tumor_bbox_pet_slice23.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

pet_4d = np.load(pet_4d_path)
bbox_info = np.load(bbox_path, allow_pickle=True).item()

last_frame = pet_4d[-1]

slice_index = bbox_info["slice_index"]
x_min = bbox_info["x_min"]
y_min = bbox_info["y_min"]
width = bbox_info["width"]
height = bbox_info["height"]

x_max = x_min + width
y_max = y_min + height

tumor_slice = last_frame[slice_index]
tumor_roi = tumor_slice[y_min:y_max, x_min:x_max]

print("=" * 60)
print("Extracting tumor ROI")
print("=" * 60)

print("PET last frame shape:", last_frame.shape)
print("Selected slice:", slice_index)

print("\nBounding box:")
print("x_min:", x_min)
print("x_max:", x_max)
print("y_min:", y_min)
print("y_max:", y_max)

print("\nTumor ROI shape:", tumor_roi.shape)
print("Tumor ROI min:", tumor_roi.min())
print("Tumor ROI max:", tumor_roi.max())
print("Tumor ROI mean:", tumor_roi.mean())

roi_path = results_dir / "tumor_roi_pet_slice23.npy"
np.save(roi_path, tumor_roi)

print("\nSaved tumor ROI:")
print(roi_path)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

vmin = np.percentile(tumor_slice, 1)
vmax = np.percentile(tumor_slice, 99.5)

axes[0].imshow(tumor_slice, cmap="gray", vmin=vmin, vmax=vmax)
axes[0].set_title("Full PET Slice 23")
axes[0].axis("off")

axes[0].plot(
    [x_min, x_max, x_max, x_min, x_min],
    [y_min, y_min, y_max, y_max, y_min],
    color="red",
    linewidth=2,
)

axes[1].imshow(tumor_roi, cmap="gray")
axes[1].set_title("Extracted Tumor ROI")
axes[1].axis("off")

figure_path = figures_dir / "tumor_roi_extracted_pet_slice23.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

print("\nStep 15 completed successfully.")