# PET-MRI Registration and Tumor Segmentation

## Overview
This project processes dynamic PET and MRI brain images, performs PET-to-MRI rigid registration, and segments a brain tumor using both classical and AI-based methods.

## Pipeline
1. DICOM loading and inspection
2. PET dynamic processing and 4D reconstruction
3. Verification of PET frame and slice ordering
4. PET visualization across time frames
5. MRI preparation
6. PET-to-MRI rigid registration
7. Numerical registration evaluation using Mutual Information
8. MIP visualization and PET-MRI fusion
9. Tumor localization using PET activity
10. Tumor segmentation:
   - PET threshold segmentation as baseline
   - MRI seeded segmentation
   - 3D SAM-based MRI tumor segmentation
11. 3D tumor mask visualization
12. PET-based numerical segmentation assessment
13. Final combined PET-MRI registration and segmentation visualization

## Structure
- `src/` → Python scripts for processing, registration, segmentation, and evaluation
- `results/figures/` → saved visual outputs
- `results/gifs/` → saved animations
- `results/*.txt` → numerical evaluation summaries

## Key Outputs
- PET slice-order verification figures
- PET-MRI registered overlay
- Mutual Information before/after registration
- 3D SAM tumor mask
- PET activity assessment of segmentation
- Final combined visualization figure

## Notes
- PET was mainly used for tumor localization and metabolic activity assessment.
- MRI was used as the anatomical reference for accurate segmentation.
- SAM was used to generate a 3D AI-based tumor segmentation across multiple MRI slices.
- PET-high activity ROI was compared with the MRI/SAM segmentation because no manual ground-truth mask was available.