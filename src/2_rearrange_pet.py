import pydicom
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "data" / "PET" / "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"

ds = pydicom.dcmread(pet_path)

pet_raw = ds.pixel_array

frame_start_times = ds[(0x0055, 0x1001)].value
frame_durations = ds[(0x0055, 0x1004)].value
frame_positions = ds[(0x0055, 0x1002)].value

num_total_frames = int(ds.NumberOfFrames)
num_time_frames = len(frame_start_times)
num_slices = num_total_frames // num_time_frames

print("=" * 60)
print("PET structure")
print("=" * 60)
print("Raw PET shape:", pet_raw.shape)
print("Total frames:", num_total_frames)
print("Number of time frames:", num_time_frames)
print("Number of slices per time frame:", num_slices)
print("Rows:", ds.Rows)
print("Columns:", ds.Columns)

print("\nExpected total frames:")
print(num_time_frames, "*", num_slices, "=", num_time_frames * num_slices)

if num_time_frames * num_slices != num_total_frames:
    raise ValueError("Time frames × slices does not match total frames.")

# Rearrange PET into:
# time, slice, row, column
pet_4d = pet_raw.reshape(num_time_frames, num_slices, ds.Rows, ds.Columns)

print("\n" + "=" * 60)
print("Rearranged PET")
print("=" * 60)
print("PET 4D shape:", pet_4d.shape)

# Also apply rescale slope if available
if (0x0055, 0x1005) in ds:
    slopes = np.array(ds[(0x0055, 0x1005)].value, dtype=np.float32)
    slopes_4d = slopes.reshape(num_time_frames, num_slices, 1, 1)

    pet_4d_scaled = pet_4d.astype(np.float32) * slopes_4d

    print("\nRescale slope found.")
    print("Scaled PET shape:", pet_4d_scaled.shape)
    print("Scaled PET min:", pet_4d_scaled.min())
    print("Scaled PET max:", pet_4d_scaled.max())
else:
    pet_4d_scaled = pet_4d.astype(np.float32)
    print("\nNo rescale slope found.")

# Save the rearranged PET 
results_dir = PROJECT_DIR / "results"
results_dir.mkdir(exist_ok=True)

output_path = results_dir / "pet_4d_scaled.npy"
np.save(output_path, pet_4d_scaled)

print("\n" + "=" * 60)
print("Saved output")
print("=" * 60)
print("Saved PET 4D file to:")
print(output_path)

print("\nStep 3 completed successfully.")