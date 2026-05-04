import pydicom
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

mri_path = PROJECT_DIR / "data" / "MRI" / "15252129_s1_AX_3D_T1__C_FSPGR_FORISI260916"
pet_path = PROJECT_DIR / "data" / "PET" / "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"

def inspect_dicom(path, name):
    print("\n" + "=" * 60)
    print(f"Reading {name}")
    print("=" * 60)

    ds = pydicom.dcmread(path)

    print("Patient ID:", ds.get("PatientID", "Not found"))
    print("Modality:", ds.get("Modality", "Not found"))
    print("Study Description:", ds.get("StudyDescription", "Not found"))
    print("Series Description:", ds.get("SeriesDescription", "Not found"))

    print("Rows:", ds.get("Rows", "Not found"))
    print("Columns:", ds.get("Columns", "Not found"))
    print("Number of Frames:", ds.get("NumberOfFrames", "Not found"))
    print("Pixel Spacing:", ds.get("PixelSpacing", "Not found"))
    print("Spacing Between Slices:", ds.get("SpacingBetweenSlices", "Not found"))

    print("Has pixel data:", "PixelData" in ds)

    if "PixelData" in ds:
        arr = ds.pixel_array
        print("Pixel array shape:", arr.shape)
        print("Pixel array dtype:", arr.dtype)
        print("Minimum pixel value:", arr.min())
        print("Maximum pixel value:", arr.max())

inspect_dicom(mri_path, "MRI")
inspect_dicom(pet_path, "PET")