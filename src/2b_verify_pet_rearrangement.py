import pydicom
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_dicom_path = PROJECT_DIR / "data" / "PET" / "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"
pet_4d_path = PROJECT_DIR / "results" / "pet_4d_scaled.npy"

results_dir = PROJECT_DIR / "results"
figures_dir = results_dir / "figures"
figures_dir.mkdir(exist_ok=True)

ds = pydicom.dcmread(pet_dicom_path)
pet_4d = np.load(pet_4d_path)

num_time_frames, num_slices, rows, cols = pet_4d.shape
num_total_frames = int(ds.NumberOfFrames)

frame_positions = np.array(ds[(0x0055, 0x1002)].value, dtype=np.float32)

# Frame Positions Vector stores x,y,z for every frame
frame_positions_xyz = frame_positions.reshape(num_total_frames, 3)

# Reshape into time, slice, xyz
frame_positions_4d = frame_positions_xyz.reshape(num_time_frames, num_slices, 3)

z_positions = frame_positions_4d[:, :, 2]

print("\nFrame positions are stored as XYZ coordinates per frame.")
print("XYZ shape:", frame_positions_xyz.shape)
print("Reshaped XYZ shape:", frame_positions_4d.shape)

print("\nFirst time-frame Z positions:")
print(z_positions[0])

print("\nLast time-frame Z positions:")
print(z_positions[-1])

same_slice_order = np.allclose(z_positions[0], z_positions[-1])
print("\nSame Z slice order in first and last time frame:", same_slice_order)

z_diffs = np.diff(z_positions[0])
print("\nZ position differences in first time frame:")
print("Min diff:", z_diffs.min())
print("Max diff:", z_diffs.max())
print("Mean diff:", z_diffs.mean())

if np.all(z_diffs > 0):
    print("Z slice order is increasing.")
elif np.all(z_diffs < 0):
    print("Z slice order is decreasing.")
else:
    print("WARNING: Z slice order is not monotonic.")