# Fetal Head Biometry via Deep Learning

Automated BPD and OFD measurement from fetal ultrasound images using landmark detection and skull segmentation.

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-orange?logo=googlecolab)

---

## Problem Statement

Measuring fetal head size during pregnancy is critical for estimating gestational age and tracking development. Clinicians manually measure two diameters on ultrasound scans:

- **BPD** (Biparietal Diameter) — skull width, side to side
- **OFD** (Occipitofrontal Diameter) — skull length, front to back

This project automates that process using deep learning — replacing manual measurement with a model that predicts the same values directly from the image.

---

## Dataset

- **Source:** HC18 Grand Challenge — publicly available fetal ultrasound dataset
- **Size:** 622 annotated images
- **Labels:** 4 landmark coordinates per image (ofd_1, ofd_2, bpd_1, bpd_2)
- **Split:** 529 train / 93 validation

---

## Two-Part Approach

### Part A — Direct Landmark Regression

Pretrained CNN backbone predicts 4 coordinate points (8 values) directly from the image.

```
Input Image (224x224)
        |
ResNet-18 / EfficientNet-B0 (pretrained backbone)
        |
Regression Head: FC(512->256) -> ReLU -> Dropout(0.3) -> FC(256->128) -> FC(128->8)
        |
[ofd_1_x, ofd_1_y, ofd_2_x, ofd_2_y, bpd_1_x, bpd_1_y, bpd_2_x, bpd_2_y]
```

### Part B — Segmentation + Ellipse Fitting

U-Net segments the fetal skull as a binary mask. BPD/OFD are then derived geometrically by fitting an ellipse to the predicted contour.

```
Input Image -> U-Net -> Binary Skull Mask -> cv2.fitEllipse() -> BPD + OFD
```

> Pseudo-masks were generated programmatically from landmark coordinates — requiring zero additional annotation effort while producing anatomically accurate skull boundaries.

---

## Scientific Experimentation Framework

Three variations were tested with exactly one variable changed at a time:

| Variation | Backbone | Loss | Augmentation | Best Val Loss | vs Baseline |
|---|---|---|---|---|---|
| V1 — Baseline | ResNet-18 | MSE | None | 0.0146 | — |
| V2 — Better Loss | ResNet-18 | Smooth L1 | None | 0.0074 | -49% |
| V3 — Augmented | EfficientNet-B0 | Smooth L1 | Albumentations | 0.0023 | -84% |

**Key finding:** Loss function choice had greater impact than backbone architecture. Domain-specific augmentation simulating probe angle, machine gain, and fetal orientation produced the largest single improvement.

---

## Results

### Clinical Measurement MAE (pixels at 224x224 resolution)

| Measurement | Part A — Direct | Part B — Segmentation | Winner |
|---|---|---|---|
| BPD | 21.3 px | 16.7 px | Part B |
| OFD | 25.8 px | 16.9 px | Part B |

Part B outperforms Part A on both measurements by reasoning about the full skull shape geometrically rather than predicting individual pixel positions.

---

## How To Run

### Train (Google Colab — T4 GPU recommended)

```bash
# Upload trainer_notebook.ipynb to Google Colab
# Runtime -> Run All
# Models save to Google Drive automatically
```

### Test on a new image

```bash
pip install torch torchvision opencv-python albumentations matplotlib

python tester.py --image path/to/ultrasound.png \
                 --landmark_weights best_V3_EfficientNet_Aug.pth \
                 --unet_weights best_unet.pth
```

Output: prediction overlay image saved as `prediction_output.png`

---

## Tech Stack

| Category | Tools |
|---|---|
| Framework | PyTorch 2.x |
| Models | ResNet-18, EfficientNet-B0, U-Net |
| Augmentation | Albumentations (keypoint-aware) |
| Image Processing | OpenCV, PIL |
| Training | Google Colab T4 GPU |
| Data | NumPy, Pandas, Matplotlib |

---

## Repository Structure

```
├── trainer_notebook.ipynb      # Full training pipeline — all 3 variations + U-Net
├── tester.py                   # Load saved models and run on new images
├── README.md
└── pathfiles/                  # Model weight checkpoints (.pth)
    ├── best_V1_ResNet_MSE.pth
    ├── best_V2_ResNet_SmoothL1.pth
    ├── best_V3_EfficientNet_Aug.pth
    └── best_unet.pth
```

---

## Future Work

- **Cascade Ensemble:** Use U-Net mask as ROI, then attention-gated regressor refines within cropped region
- **Heatmap Detection:** Predict 2D Gaussian heatmaps per landmark instead of direct coordinates
- **Clinical Deployment:** DICOM PixelSpacing metadata for pixel-to-mm conversion and Hadlock chart comparison
- **Uncertainty Estimation:** Monte Carlo Dropout for confidence scoring and human-review flagging

---

## Author

**Apoorva**  
B.E. Electronics & Communication Engineering  
BMS Institute of Technology & Management, Bengaluru (2023–2027)  
SIH 2024 National Winner
