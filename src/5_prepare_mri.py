import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "data" / "MRI" / "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

ds = pydicom.dcmread(mri_path)

mri = ds.pixel_array.astype(np.float32)

print("=" * 60)
print("Loaded MRI")
print("=" * 60)
print("MRI shape:", mri.shape)
print("MRI format: slice, row, column")
print("MRI dtype:", mri.dtype)
print("MRI min:", mri.min())
print("MRI max:", mri.max())

print("\nMRI DICOM spacing information")
print("Pixel Spacing:", ds.get("PixelSpacing", "Not found"))
print("Spacing Between Slices:", ds.get("SpacingBetweenSlices", "Not found"))
print("Number of Frames:", ds.get("NumberOfFrames", "Not found"))

# Save MRI as NumPy file for later registration
output_path = results_dir / "mri.npy"
np.save(output_path, mri)

print("\nSaved MRI file:")
print(output_path)

# Visualize middle slice
middle_slice = mri.shape[0] // 2
mri_middle = mri[middle_slice]

plt.figure(figsize=(6, 6))
plt.imshow(mri_middle, cmap="gray")
plt.title(f"MRI Middle Slice - Slice {middle_slice}")
plt.axis("off")
plt.colorbar()
plt.savefig(figures_dir / "mri_middle_slice.png", dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figures_dir / "mri_middle_slice.png")

print("\nStep 6 completed successfully.")