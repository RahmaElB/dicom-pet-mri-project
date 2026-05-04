import numpy as np
import SimpleITK as sitk
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
pet_avg_path = PROJECT_DIR / "results" / "pet_average.npy"

print("=" * 60)
print("Checking registration setup")
print("=" * 60)

mri = np.load(mri_path)
pet = np.load(pet_avg_path)

print("MRI shape:", mri.shape)
print("PET average shape:", pet.shape)

print("\nSimpleITK version:")
print(sitk.Version())

# Convert NumPy arrays to SimpleITK images
mri_img = sitk.GetImageFromArray(mri)
pet_img = sitk.GetImageFromArray(pet)

# Set voxel spacing
# MRI spacing: z=1.0, y=1.0, x=1.0
# PET spacing: z=3.27, y=1.171875, x=1.171875
mri_img.SetSpacing((1.0, 1.0, 1.0))
pet_img.SetSpacing((1.171875, 1.171875, 3.27))

print("\nMRI SimpleITK size:", mri_img.GetSize())
print("MRI SimpleITK spacing:", mri_img.GetSpacing())

print("\nPET SimpleITK size:", pet_img.GetSize())
print("PET SimpleITK spacing:", pet_img.GetSpacing())

print("\nStep 8 completed successfully.")