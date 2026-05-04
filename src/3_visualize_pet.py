import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"
results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

pet_4d = np.load(pet_path)

print("=" * 60)
print("Loaded PET 4D")
print("=" * 60)
print("PET shape:", pet_4d.shape)
print("Format: time, slice, row, column")

num_time_frames, num_slices, rows, cols = pet_4d.shape

middle_slice = num_slices // 2

last_frame = pet_4d[-1]
average_frame = pet_4d.mean(axis=0)

last_middle_slice = last_frame[middle_slice]
average_middle_slice = average_frame[middle_slice]

print("Number of time frames:", num_time_frames)
print("Number of slices:", num_slices)
print("Middle slice index:", middle_slice)

# Save last frame middle slice
plt.figure(figsize=(6, 6))
plt.imshow(last_middle_slice, cmap="gray")
plt.title("PET Last Time Frame - Middle Slice")
plt.axis("off")
plt.colorbar()
plt.savefig(figures_dir / "pet_last_frame_middle_slice.png", dpi=300, bbox_inches="tight")
plt.show()

# Save average frame middle slice
plt.figure(figsize=(6, 6))
plt.imshow(average_middle_slice, cmap="gray")
plt.title("PET Average of All Time Frames - Middle Slice")
plt.axis("off")
plt.colorbar()
plt.savefig(figures_dir / "pet_average_frame_middle_slice.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figures:")
print(figures_dir / "pet_last_frame_middle_slice.png")
print(figures_dir / "pet_average_frame_middle_slice.png")

print("\nStep 4 completed successfully.")