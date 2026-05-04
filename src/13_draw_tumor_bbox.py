import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_4d_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

pet_4d = np.load(pet_4d_path)

last_frame = pet_4d[-1]

tumor_slice_index = 23
tumor_slice = last_frame[tumor_slice_index]

print("=" * 60)
print("Tumor bounding box setup")
print("=" * 60)
print("PET last frame shape:", last_frame.shape)
print("Selected tumor slice:", tumor_slice_index)

# Estimated bounding box on slice 23
# Format:
# x_min, y_min, width, height
x_min = 130
y_min = 120
width = 70
height = 70

x_center = x_min + width / 2
y_center = y_min + height / 2

print("\nEstimated tumor bounding box:")
print("x_min:", x_min)
print("y_min:", y_min)
print("width:", width)
print("height:", height)
print("x_center:", x_center)
print("y_center:", y_center)

vmin = np.percentile(tumor_slice, 1)
vmax = np.percentile(tumor_slice, 99.5)

fig, ax = plt.subplots(figsize=(7, 7))

ax.imshow(tumor_slice, cmap="gray", vmin=vmin, vmax=vmax)

rect = patches.Rectangle(
    (x_min, y_min),
    width,
    height,
    linewidth=2,
    edgecolor="red",
    facecolor="none"
)

ax.add_patch(rect)

ax.plot(x_center, y_center, "bo", markersize=6)

ax.set_title(f"Tumor Bounding Box - PET Last Frame Slice {tumor_slice_index}")
ax.axis("off")

figure_path = figures_dir / "tumor_bbox_pet_slice23.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

# Save coordinates 
bbox_info = {
    "slice_index": tumor_slice_index,
    "x_min": x_min,
    "y_min": y_min,
    "width": width,
    "height": height,
    "x_center": x_center,
    "y_center": y_center,
}

bbox_path = results_dir / "tumor_bbox_pet_slice23.npy"
np.save(bbox_path, bbox_info)

print("\nSaved bounding box info:")
print(bbox_path)

print("\nStep 14 completed successfully.")