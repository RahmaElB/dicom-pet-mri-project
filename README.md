## PET-MRI Registration and Tumor Segmentation

### Overview
This project processes dynamic PET and MRI images, performs rigid registration, and segments a brain tumor using both classical and AI-based methods.

### Pipeline
1. DICOM loading and inspection
2. PET dynamic processing (4D reconstruction)
3. PET visualization (frames, averages, GIF)
4. MRI preparation
5. PET-MRI rigid registration
6. MIP visualization and rotating GIF
7. Tumor localization (PET)
8. Tumor segmentation:
   - PET threshold (baseline)
   - MRI seeded segmentation
   - SAM (AI-based)

### Structure
- `src/` → all code files
- `results/` → figures, GIFs, outputs

### Notes
- PET used for tumor localization
- MRI used for accurate segmentation
- SAM used for AI-based segmentation
