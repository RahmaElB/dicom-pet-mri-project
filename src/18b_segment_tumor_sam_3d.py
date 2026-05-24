import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamPredictor
from scipy.ndimage import binary_closing, binary_fill_holes

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
checkpoint_path = PROJECT_DIR / "models" / "sam_vit_b_01ec64.pth"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)

print("=" * 60)
print("3D SAM tumor segmentation")
print("=" * 60)
print("MRI shape:", mri.shape)

# Tumor slice range around the manually identified lesion
slice_start = 72
slice_end = 80 

# Bounding box around lesion in MRI coordinates
x_min = 145
x_max = 195
y_min = 155
y_max = 215

print("Segmenting slices:", slice_start, "to", slice_end - 1)
print("Bounding box:", x_min, y_min, x_max, y_max)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
sam.to(device=device)

predictor = SamPredictor(sam)

mask_3d = np.zeros_like(mri, dtype=bool)

box = np.array([x_min, y_min, x_max, y_max])

slice_scores = {}

for slice_index in range(slice_start, slice_end):
    print(f"\nProcessing slice {slice_index}")

    mri_slice = mri[slice_index]

    # Normalize MRI slice to 0-255 RGB for SAM
    mri_norm = (mri_slice - np.percentile(mri_slice, 1)) / (
        np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1) + 1e-8
    )
    mri_norm = np.clip(mri_norm, 0, 1)
    mri_uint8 = (mri_norm * 255).astype(np.uint8)
    mri_rgb = np.stack([mri_uint8] * 3, axis=-1)

    predictor.set_image(mri_rgb)

    masks, scores, logits = predictor.predict(
        box=box,
        multimask_output=True
    )

    best_idx = int(np.argmax(scores))
    mask = masks[best_idx]

    # Clean each 2D mask slightly
    mask = binary_closing(mask, iterations=1)
    mask = binary_fill_holes(mask)

    mask_3d[slice_index] = mask

    slice_scores[slice_index] = float(scores[best_idx])

    print("SAM scores:", scores)
    print("Selected mask:", best_idx)
    print("Selected score:", scores[best_idx])
    print("Mask pixels:", mask.sum())

# Save 3D mask
mask_path = results_dir / "tumor_mask_sam_3d.npy"
np.save(mask_path, mask_3d)

print("\nSaved 3D SAM mask:")
print(mask_path)

print("\n3D mask summary")
print("Total mask voxels:", mask_3d.sum())
print("Segmented slices:", np.where(mask_3d.sum(axis=(1, 2)) > 0)[0])

# Save text summary
summary_path = results_dir / "tumor_mask_sam_3d_summary.txt"
with open(summary_path, "w") as f:
    f.write("3D SAM tumor segmentation summary\n")
    f.write("=================================\n")
    f.write(f"Slice range: {slice_start} to {slice_end - 1}\n")
    f.write(f"Bounding box: [{x_min}, {y_min}, {x_max}, {y_max}]\n")
    f.write(f"Total mask voxels: {int(mask_3d.sum())}\n")
    f.write("Segmented slices:\n")
    for z in np.where(mask_3d.sum(axis=(1, 2)) > 0)[0]:
        f.write(f"Slice {z}: voxels={int(mask_3d[z].sum())}, SAM score={slice_scores.get(int(z), None)}\n")

print("Saved summary:")
print(summary_path)

# Visualization: selected slices
selected_slices = [72, 73, 74, 75, 76, 77, 78, 79]

fig, axes = plt.subplots(len(selected_slices), 3, figsize=(12, 4 * len(selected_slices)))

for row, slice_index in enumerate(selected_slices):
    mri_slice = mri[slice_index]
    mask_slice = mask_3d[slice_index]

    mri_norm = (mri_slice - np.percentile(mri_slice, 1)) / (
        np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1) + 1e-8
    )
    mri_norm = np.clip(mri_norm, 0, 1)

    axes[row, 0].imshow(mri_norm, cmap="gray")
    axes[row, 0].set_title(f"MRI Slice {slice_index}")
    rect = patches.Rectangle(
        (x_min, y_min),
        x_max - x_min,
        y_max - y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
    )
    axes[row, 0].add_patch(rect)

    axes[row, 1].imshow(mask_slice, cmap="gray")
    axes[row, 1].set_title("3D SAM Mask Slice")

    axes[row, 2].imshow(mri_norm, cmap="gray")
    axes[row, 2].imshow(mask_slice, cmap="Reds", alpha=0.45)
    axes[row, 2].set_title("Mask Overlay")

    for col in range(3):
        axes[row, col].axis("off")

plt.tight_layout()

figure_path = figures_dir / "sam_3d_segmentation_selected_slices.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved selected-slices visualization:")
print(figure_path)

print("\nStep completed successfully.")