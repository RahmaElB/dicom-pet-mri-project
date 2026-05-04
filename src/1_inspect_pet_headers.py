import pydicom
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]

pet_path = PROJECT_DIR / "data" / "PET" / "02324177_s2_e_1_BRAIN_DINAMIC_COLINA_AC_FORISI260916"

ds = pydicom.dcmread(pet_path)

print("=" * 60)
print("Basic PET information")
print("=" * 60)

print("Rows:", ds.Rows)
print("Columns:", ds.Columns)
print("Number of Frames:", ds.NumberOfFrames)
print("Pixel array shape:", ds.pixel_array.shape)

print("\n" + "=" * 60)
print("Professor-mentioned PET headers")
print("=" * 60)

headers_to_check = [
    ("Number of Frames", (0x0028, 0x0008)),
    ("Rows", (0x0028, 0x0010)),
    ("Columns", (0x0028, 0x0011)),
    ("Spacing Between Slices", (0x0018, 0x0088)),
    ("Pixel Spacing", (0x0028, 0x0030)),
    ("Frame Positions Vector", (0x0055, 0x1002)),
    ("Frame Start Times Vector", (0x0055, 0x1001)),
    ("Frame Durations Vector", (0x0055, 0x1004)),
]

for name, tag in headers_to_check:
    print("\n" + "-" * 60)
    print(name, tag)

    if tag in ds:
        value = ds[tag].value

        if hasattr(value, "__len__") and not isinstance(value, str):
            print("Length:", len(value))
            print("First 10 values:", value[:10])
            print("Last 10 values:", value[-10:])
        else:
            print("Value:", value)
    else:
        print("NOT FOUND")

print("\n" + "=" * 60)
print("All private 0055 tags")
print("=" * 60)

for elem in ds:
    if elem.tag.group == 0x0055:
        value = elem.value

        print("\nTag:", elem.tag)
        print("Name:", elem.name)

        if hasattr(value, "__len__") and not isinstance(value, str):
            print("Length:", len(value))
            print("First 10 values:", value[:10])
        else:
            print("Value:", value)