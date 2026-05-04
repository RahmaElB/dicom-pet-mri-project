import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
registered_pet_path = PROJECT_DIR / "results" / "pet_registered_to_mri.npy"
bbox_path = PROJECT_DIR / "results" / "tumor_bbox_pet_slice23.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)
pet_registered = np.load(registered_pet_path).astype(np.float32)
bbox_info = np.load(bbox_path, allow_pickle=True).item()

print("=" * 60)
print("Mapping PET tumor bounding box to MRI space")
print("=" * 60)

print("MRI shape:", mri.shape)
print("Registered PET shape:", pet_registered.shape)
print("Original PET bbox info:", bbox_info)

if mri.shape != pet_registered.shape:
    raise ValueError("MRI and registered PET must have the same shape.")

# Original PET bbox
pet_slice_index = int(bbox_info["slice_index"])
x_min = int(bbox_info["x_min"])
y_min = int(bbox_info["y_min"])
width = int(bbox_info["width"])
height = int(bbox_info["height"])

x_max = x_min + width
y_max = y_min + height

z_max, rows, cols = mri.shape

# Clip box
x_min = max(0, min(x_min, cols - 1))
x_max = max(0, min(x_max, cols))
y_min = max(0, min(y_min, rows - 1))
y_max = max(0, min(y_max, rows))

width = x_max - x_min
height = y_max - y_min

x_center = x_min + width / 2
y_center = y_min + height / 2

# Manually selected MRI slice where the lesion is clearly visible
mri_slice_index = 78

print("\nManual MRI slice selection")
print("Original PET tumor slice:", pet_slice_index)
print("Selected MRI slice:", mri_slice_index)

print("\nMRI-space tumor box estimate:")
print("slice_index:", mri_slice_index)
print("x_min:", x_min)
print("x_max:", x_max)
print("y_min:", y_min)
print("y_max:", y_max)
print("width:", width)
print("height:", height)
print("x_center:", x_center)
print("y_center:", y_center)

mri_slice = mri[mri_slice_index]
pet_slice = pet_registered[mri_slice_index]

mri_norm = (mri_slice - np.percentile(mri_slice, 1)) / (
    np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1) + 1e-8
)
mri_norm = np.clip(mri_norm, 0, 1)

pet_norm = (pet_slice - np.percentile(pet_slice, 1)) / (
    np.percentile(pet_slice, 99.5) - np.percentile(pet_slice, 1) + 1e-8
)
pet_norm = np.clip(pet_norm, 0, 1)

mri_bbox_info = {
    "slice_index": int(mri_slice_index),
    "x_min": int(x_min),
    "y_min": int(y_min),
    "x_max": int(x_max),
    "y_max": int(y_max),
    "width": int(width),
    "height": int(height),
    "x_center": float(x_center),
    "y_center": float(y_center),
}

mri_bbox_path = results_dir / "tumor_bbox_mri.npy"
np.save(mri_bbox_path, mri_bbox_info)

print("\nSaved MRI bbox info:")
print(mri_bbox_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax in axes:
    ax.axis("off")

axes[0].imshow(mri_norm, cmap="gray")
axes[0].set_title(f"MRI Slice {mri_slice_index} + Tumor Box")
rect0 = patches.Rectangle(
    (x_min, y_min),
    width,
    height,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
axes[0].add_patch(rect0)
axes[0].plot(x_center, y_center, "bo", markersize=5)

axes[1].imshow(pet_norm, cmap="hot")
axes[1].set_title(f"Registered PET Slice {mri_slice_index} + Box")
rect1 = patches.Rectangle(
    (x_min, y_min),
    width,
    height,
    linewidth=2,
    edgecolor="cyan",
    facecolor="none",
)
axes[1].add_patch(rect1)
axes[1].plot(x_center, y_center, "bo", markersize=5)

axes[2].imshow(mri_norm, cmap="gray")
axes[2].imshow(pet_norm, cmap="hot", alpha=0.4)
axes[2].set_title("MRI + Registered PET + Box")
rect2 = patches.Rectangle(
    (x_min, y_min),
    width,
    height,
    linewidth=2,
    edgecolor="lime",
    facecolor="none",
)
axes[2].add_patch(rect2)
axes[2].plot(x_center, y_center, "bo", markersize=5)

plt.tight_layout()

figure_path = figures_dir / "tumor_bbox_mapped_to_mri.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

print("\nStep 17 completed successfully.")