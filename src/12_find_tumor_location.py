import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_4d_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

pet_4d = np.load(pet_4d_path)

print("=" * 60)
print("Loaded PET 4D")
print("=" * 60)
print("PET shape:", pet_4d.shape)

# Use last PET time frame because tumor is clearest there
last_frame = pet_4d[-1]

print("Last frame shape:", last_frame.shape)

# Show all slices so we can inspect where tumor appears
num_slices = last_frame.shape[0]

fig, axes = plt.subplots(7, 7, figsize=(14, 14))
axes = axes.ravel()

vmin = np.percentile(last_frame, 1)
vmax = np.percentile(last_frame, 99.5)

for i in range(49):
    axes[i].axis("off")

    if i < num_slices:
        axes[i].imshow(last_frame[i], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Slice {i}", fontsize=8)

plt.suptitle("PET Last Frame - All Slices")
plt.tight_layout()

output_path = figures_dir / "pet_last_frame_all_slices.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved slice overview:")
print(output_path)

print("\nStep 13 completed successfully.")