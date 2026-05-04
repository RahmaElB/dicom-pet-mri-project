import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"

results_dir = PROJECT_DIR / "results"
gifs_dir = results_dir / "gifs"
gifs_dir.mkdir(exist_ok=True)

pet_4d = np.load(pet_path)

print("=" * 60)
print("Loaded PET 4D")
print("=" * 60)
print("PET shape:", pet_4d.shape)
print("Format: time, slice, row, column")

num_time_frames, num_slices, rows, cols = pet_4d.shape

middle_slice = num_slices // 2
middle_row = rows // 2
middle_col = cols // 2

print("Number of time frames:", num_time_frames)
print("Middle axial slice:", middle_slice)
print("Middle coronal row:", middle_row)
print("Middle sagittal column:", middle_col)

# Use a stable intensity range so the GIF does not flicker
vmin = np.percentile(pet_4d, 1)
vmax = np.percentile(pet_4d, 99.5)

print("Display vmin:", vmin)
print("Display vmax:", vmax)

frames = []

for t in range(num_time_frames):
    volume = pet_4d[t]

    axial = volume[middle_slice, :, :]
    coronal = volume[:, middle_row, :]
    sagittal = volume[:, :, middle_col]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(axial, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("Axial")

    axes[1].imshow(coronal, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    axes[1].set_title("Coronal")

    axes[2].imshow(sagittal, cmap="gray", vmin=vmin, vmax=vmax, aspect="auto")
    axes[2].set_title("Sagittal")

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"PET Dynamic Frame {t + 1}/{num_time_frames}")
    plt.tight_layout()

    # Convert matplotlib figure to image array
    fig.canvas.draw()

    rgba_image = np.asarray(fig.canvas.buffer_rgba())
    image = rgba_image[:, :, :3]

    frames.append(image)

    plt.close(fig)

gif_path = gifs_dir / "pet_three_median_planes.gif"

imageio.mimsave(gif_path, frames, duration=0.4)

print("\nSaved GIF:")
print(gif_path)

print("\nStep 5 completed successfully.")