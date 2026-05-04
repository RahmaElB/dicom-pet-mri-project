import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import rotate
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
pet_path = PROJECT_DIR / "results" / "pet_registered_to_mri.npy"

results_dir = PROJECT_DIR / "results"
gifs_dir = results_dir / "gifs"
gifs_dir.mkdir(exist_ok=True)

mri = np.load(mri_path).astype(np.float32)
pet = np.load(pet_path).astype(np.float32)

print("=" * 60)
print("Loaded registered images")
print("=" * 60)
print("MRI shape:", mri.shape)
print("PET shape:", pet.shape)

if mri.shape != pet.shape:
    raise ValueError("MRI and PET must have same shape.")

# Normalize
mri_norm = (mri - np.percentile(mri, 5)) / (
    np.percentile(mri, 95) - np.percentile(mri, 5) + 1e-8
)
mri_norm = np.clip(mri_norm, 0, 1)

pet_norm = (pet - np.percentile(pet, 1)) / (
    np.percentile(pet, 99.5) - np.percentile(pet, 1) + 1e-8
)
pet_norm = np.clip(pet_norm, 0, 1)

frames = []

angles = range(0, 360, 10)

for angle in angles:
    print(f"Creating frame angle {angle} degrees")

    # Rotate around axial axis, then project
    mri_rot = rotate(mri_norm, angle, axes=(1, 2), reshape=False, order=1)
    pet_rot = rotate(pet_norm, angle, axes=(1, 2), reshape=False, order=1)

    # MRI mean projection gives anatomy; PET max projection gives uptake
    mri_proj = np.mean(mri_rot, axis=0)
    pet_proj = np.max(pet_rot, axis=0)

    fig, axes_plot = plt.subplots(1, 3, figsize=(15, 5))

    axes_plot[0].imshow(mri_proj, cmap="gray")
    axes_plot[0].set_title("MRI Reference")
    axes_plot[0].axis("off")

    axes_plot[1].imshow(pet_proj, cmap="hot")
    axes_plot[1].set_title("Registered PET")
    axes_plot[1].axis("off")

    axes_plot[2].imshow(mri_proj, cmap="gray")
    axes_plot[2].imshow(pet_proj, cmap="hot", alpha=0.45)
    axes_plot[2].set_title("MRI + PET Alpha Fusion")
    axes_plot[2].axis("off")

    fig.suptitle(f"Rotating MIP - Angle {angle}°")
    plt.tight_layout()

    fig.canvas.draw()
    rgba_image = np.asarray(fig.canvas.buffer_rgba())
    image = rgba_image[:, :, :3]

    frames.append(image)
    plt.close(fig)

gif_path = gifs_dir / "rotating_mip_mri_pet_fusion.gif"
imageio.mimsave(gif_path, frames, duration=0.25)

print("\nSaved rotating MIP GIF:")
print(gif_path)

print("\nStep 12 completed successfully.")