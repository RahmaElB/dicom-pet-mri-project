import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import torch
from segment_anything import sam_model_registry, SamPredictor

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
checkpoint_path = PROJECT_DIR / "models" / "sam_vit_b_01ec64.pth"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)

slice_index = 78

# Final box around lesion
x_min = 145
x_max = 195
y_min = 155
y_max = 215

mri_slice = mri[slice_index]

# Normalize MRI slice to 0-255 RGB for SAM
mri_norm = (mri_slice - np.percentile(mri_slice, 1)) / (
    np.percentile(mri_slice, 99) - np.percentile(mri_slice, 1) + 1e-8
)
mri_norm = np.clip(mri_norm, 0, 1)
mri_uint8 = (mri_norm * 255).astype(np.uint8)
mri_rgb = np.stack([mri_uint8] * 3, axis=-1)

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint_path))
sam.to(device=device)

predictor = SamPredictor(sam)
predictor.set_image(mri_rgb)

box = np.array([x_min, y_min, x_max, y_max])

masks, scores, logits = predictor.predict(
    box=box,
    multimask_output=True
)

best_idx = int(np.argmax(scores))
mask = masks[best_idx]

print("SAM scores:", scores)
print("Selected mask:", best_idx)
print("Mask pixels:", mask.sum())

np.save(results_dir / "tumor_mask_sam_mri_slice78.npy", mask)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax in axes:
    ax.axis("off")

axes[0].imshow(mri_slice, cmap="gray")
axes[0].set_title("MRI + Box Prompt")
rect = patches.Rectangle(
    (x_min, y_min),
    x_max - x_min,
    y_max - y_min,
    linewidth=2,
    edgecolor="red",
    facecolor="none",
)
axes[0].add_patch(rect)

axes[1].imshow(mask, cmap="gray")
axes[1].set_title("SAM Tumor Mask")

axes[2].imshow(mri_slice, cmap="gray")
axes[2].imshow(mask, cmap="Reds", alpha=0.45)
axes[2].set_title("SAM Mask Overlay on MRI")

plt.tight_layout()

figure_path = figures_dir / "sam_mri_tumor_segmentation.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", figure_path)