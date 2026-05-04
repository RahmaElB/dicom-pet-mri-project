import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "results" / "mri.npy"
pet_path = PROJECT_DIR / "results" / "pet_average.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Loading MRI and PET average")
print("=" * 60)

mri = np.load(mri_path).astype(np.float32)
pet = np.load(pet_path).astype(np.float32)

print("MRI shape:", mri.shape)
print("PET shape:", pet.shape)

fixed_img = sitk.GetImageFromArray(mri)
moving_img = sitk.GetImageFromArray(pet)

fixed_img.SetSpacing((1.0, 1.0, 1.0))
moving_img.SetSpacing((1.171875, 1.171875, 3.27))

print("\nFixed MRI size:", fixed_img.GetSize())
print("Fixed MRI spacing:", fixed_img.GetSpacing())

print("\nMoving PET size:", moving_img.GetSize())
print("Moving PET spacing:", moving_img.GetSpacing())

print("\n" + "=" * 60)
print("Initializing rigid transform")
print("=" * 60)

initial_transform = sitk.CenteredTransformInitializer(
    fixed_img,
    moving_img,
    sitk.Euler3DTransform(),
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)

print("Initial transform:")
print(initial_transform)

print("\n" + "=" * 60)
print("Running registration")
print("=" * 60)

registration = sitk.ImageRegistrationMethod()

registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

registration.SetMetricSamplingStrategy(registration.RANDOM)
registration.SetMetricSamplingPercentage(0.05)

registration.SetInterpolator(sitk.sitkLinear)

registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=2.0,
    minStep=1e-4,
    numberOfIterations=200,
    gradientMagnitudeTolerance=1e-8
)

registration.SetOptimizerScalesFromPhysicalShift()

registration.SetInitialTransform(initial_transform, inPlace=False)

final_transform = registration.Execute(fixed_img, moving_img)

print("Final metric value:", registration.GetMetricValue())
print("Optimizer stop condition:")
print(registration.GetOptimizerStopConditionDescription())

print("\nFinal transform:")
print(final_transform)

print("\n" + "=" * 60)
print("Resampling registered PET into MRI space")
print("=" * 60)

registered_pet_img = sitk.Resample(
    moving_img,
    fixed_img,
    final_transform,
    sitk.sitkLinear,
    0.0,
    moving_img.GetPixelID()
)

registered_pet = sitk.GetArrayFromImage(registered_pet_img).astype(np.float32)

print("Registered PET shape:", registered_pet.shape)
print("Registered PET min:", registered_pet.min())
print("Registered PET max:", registered_pet.max())

output_path = results_dir / "pet_registered_to_mri.npy"
np.save(output_path, registered_pet)

transform_path = results_dir / "pet_to_mri_rigid_transform.tfm"
sitk.WriteTransform(final_transform, str(transform_path))

print("\nSaved registered PET:")
print(output_path)

print("\nSaved transform:")
print(transform_path)

print("\n" + "=" * 60)
print("Creating visual comparison")
print("=" * 60)

middle_slice = fixed_img.GetSize()[2] // 2

mri_slice = mri[middle_slice]
registered_pet_slice = registered_pet[middle_slice]

# Normalize for overlay
mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
pet_norm = (registered_pet_slice - registered_pet_slice.min()) / (
    registered_pet_slice.max() - registered_pet_slice.min() + 1e-8
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(mri_slice, cmap="gray")
axes[0].set_title("MRI Reference")
axes[0].axis("off")

axes[1].imshow(registered_pet_slice, cmap="hot")
axes[1].set_title("Registered PET")
axes[1].axis("off")

axes[2].imshow(mri_norm, cmap="gray")
axes[2].imshow(pet_norm, cmap="hot", alpha=0.4)
axes[2].set_title("MRI + Registered PET Overlay")
axes[2].axis("off")

plt.tight_layout()
figure_path = figures_dir / "registered_pet_mri_overlay.png"
plt.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:")
print(figure_path)

print("\nStep 10 completed successfully.")