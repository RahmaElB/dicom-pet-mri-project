import numpy as np
import SimpleITK as sitk
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
pet_average_path = PROJECT_DIR / "results" / "pet_average.npy"
pet_registered_path = PROJECT_DIR / "results" / "pet_registered_to_mri.npy"

results_dir = PROJECT_DIR / "results"

mri = np.load(mri_path).astype(np.float32)
pet_average = np.load(pet_average_path).astype(np.float32)
pet_registered = np.load(pet_registered_path).astype(np.float32)

print("=" * 60)
print("Registration numerical evaluation")
print("=" * 60)

print("MRI shape:", mri.shape)
print("PET average shape:", pet_average.shape)
print("Registered PET shape:", pet_registered.shape)


def normalize_image(image):
    p1 = np.percentile(image, 1)
    p99 = np.percentile(image, 99)
    image_norm = (image - p1) / (p99 - p1 + 1e-8)
    image_norm = np.clip(image_norm, 0, 1)
    return image_norm.astype(np.float32)


def compute_mutual_information(img1, img2, bins=64):
    """
    Computes histogram-based mutual information between two 3D images.
    Higher MI means stronger statistical dependency between both images.
    """
    img1 = normalize_image(img1)
    img2 = normalize_image(img2)

    img1_flat = img1.ravel()
    img2_flat = img2.ravel()

    hist_2d, _, _ = np.histogram2d(img1_flat, img2_flat, bins=bins)

    joint_prob = hist_2d / np.sum(hist_2d)

    p1 = np.sum(joint_prob, axis=1)
    p2 = np.sum(joint_prob, axis=0)

    p1_p2 = p1[:, None] * p2[None, :]

    nonzero = joint_prob > 0

    mi = np.sum(joint_prob[nonzero] * np.log(joint_prob[nonzero] / p1_p2[nonzero]))

    return mi


# For fair comparison, create an initial PET image in MRI space using identity transform
mri_img = sitk.GetImageFromArray(mri)
pet_img = sitk.GetImageFromArray(pet_average)

mri_img.SetSpacing((1.0, 1.0, 1.0))
pet_img.SetSpacing((1.171875, 1.171875, 3.27))

identity_transform = sitk.Transform(3, sitk.sitkIdentity)

pet_initial_img = sitk.Resample(
    pet_img,
    mri_img,
    identity_transform,
    sitk.sitkLinear,
    0.0,
    pet_img.GetPixelID()
)

pet_initial = sitk.GetArrayFromImage(pet_initial_img).astype(np.float32)

print("\nInitial PET in MRI space shape:", pet_initial.shape)

mi_before = compute_mutual_information(mri, pet_initial)
mi_after = compute_mutual_information(mri, pet_registered)

print("\nMutual Information before registration:", mi_before)
print("Mutual Information after registration:", mi_after)

if mi_after > mi_before:
    print("Result: MI improved after registration.")
else:
    print("Result: MI did not improve. Check registration visually and parameters.")

improvement = ((mi_after - mi_before) / (abs(mi_before) + 1e-8)) * 100
print("Relative MI change (%):", improvement)

# Save results as text file
output_path = results_dir / "registration_mutual_information.txt"

with open(output_path, "w") as f:
    f.write("Registration numerical evaluation\n")
    f.write("=================================\n")
    f.write(f"Mutual Information before registration: {mi_before:.6f}\n")
    f.write(f"Mutual Information after registration: {mi_after:.6f}\n")
    f.write(f"Relative MI change (%): {improvement:.2f}\n")

print("\nSaved MI evaluation to:")
print(output_path)

print("\nStep completed successfully.")