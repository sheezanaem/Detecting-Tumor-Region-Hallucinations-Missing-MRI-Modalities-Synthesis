# Detecting-Tumor-Region-Hallucinations-Missing-MRI-Modalities-Synthesis
Pipeline for data preparation, multi-modal MRI synthesis, and detecting tumor-region hallucinations across missing BraTS 2021 brain MRI modalities (FLAIR, T1, T1ce, T2) using ground-truth tumor segmentation masks.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Dataset: BraTS 2021](https://img.shields.io/badge/Dataset-BraTS%202021-green.svg)](https://www.med.upenn.edu/cbica/brats2021/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A reproducible pipeline for environment setup, 3D multi-modal MRI extraction, missing-modality synthesis, and quantitative detection of **tumor-region hallucinations** on the **BraTS 2021** brain tumor dataset.

---

## Table of Contents
- [Overview](#overview)
- [Motivation & Clinical Problem](#motivation--clinical-problem)
- [Dataset Details](#dataset-details)
- [Repository Structure](#repository-structure)
- [Installation & Offline Wheel Setup](#installation--offline-wheel-setup)
- [Data Preparation Workflow](#data-preparation-workflow)
- [Hallucination Evaluation Framework](#hallucination-evaluation-framework)
- [Citation & Acknowledgments](#citation--acknowledgments)

---

## Overview
In clinical neuroimaging, acquiring all four standard structural MRI sequences—**FLAIR**, **T1**, **T1ce** (contrast-enhanced T1), and **T2**—is often hindered by scan time constraints, motion artifacts, or contrast agent contraindications. While deep generative models (e.g., GANs, Diffusion Models, and Vision Transformers) can synthesize missing modalities, they are prone to **structural hallucinations**: fabricating artificial tumor enhancements in healthy tissue or erasing true lesion boundaries.

This repository provides:
1. **Air-Gapped & Standard Environment Setup:** Automated installation and verification of deep learning, medical imaging (`nibabel`, `pydicom`), and evaluation dependencies via local `.whl` archives or `pip`.
2. **Automated Data Extraction:** Scripts to unpack and organize multi-modal 3D NIfTI volumes from BraTS 2021 Task 1 alongside RSNA-MICCAI radiogenomic reference scans.
3. **Region-Stratified Evaluation:** Tools to isolate tumor sub-regions using ground-truth segmentation masks (`seg`) to measure synthesis fidelity and hallucination rates separately inside and outside pathological tissue.

---

## Motivation & Clinical Problem
Standard whole-brain image quality metrics (global PSNR, global SSIM) are dominated by background and healthy brain parenchyma, masking critical synthesis errors inside small tumor sub-regions. By pairing synthesized MRI volumes with ground-truth segmentation masks, this pipeline enables localized auditing of:
- **False-Positive Hallucinations:** Hyperintense or contrast-like artifacts generated in normal-appearing brain tissue.
- **False-Negative Omissions:** Loss of necrotic core, peritumoral edema, or enhancing tumor visibility in imputed sequences.

---

## Dataset Details

### BraTS 2021 Task 1 (Primary)
Each subject directory contains five co-registered, skull-stripped 3D NIfTI volumes (`float32`) with spatial dimensions $240 \times 240 \times 155$ at $1\text{ mm}^3$ isotropic resolution:

| File Suffix | Modality / Role | Clinical Significance |
| :--- | :--- | :--- |
| `*_flair.nii.gz` | Fluid-Attenuated Inversion Recovery | Highlights peritumoral edema and infiltrative lesions |
| `*_t1.nii.gz` | Native T1-weighted | Captures baseline anatomical structure |
| `*_t1ce.nii.gz` | Contrast-Enhanced T1-weighted | Delineates active blood-brain barrier breakdown (enhancing tumor) |
| `*_t2.nii.gz` | T2-weighted | Emphasizes fluid content and edema boundaries |
| `*_seg.nii.gz` | Ground-Truth Segmentation Mask | Voxel-wise labels: `0` (Background), `1` (NCR/NET), `2` (ED), `4` (ET) |

### RSNA-MICCAI Radiogenomic Dataset (Reference Comparison)
Includes utility loaders for comparing DICOM and PNG slice representations against BraTS 3D NIfTI volumes.

---

## Repository Structure

```text
Detecting-Tumor-Region-Hallucinations-Missing-MRI-Modalities-Synthesis/
├── notebooks/
│   └── 01_setup_and_brats2021_data_loading.ipynb   # Environment check, .tar extraction, NIfTI inspection
├── src/
│   ├── data/                                       # 3D NIfTI loaders, slice extractors, modality dropout
│   ├── models/                                     # Missing-modality synthesis architectures
│   └── metrics/                                    # Tumor-masked PSNR, SSIM, LPIPS, and segmentation consistency
├── wheels/                                         # (Optional, gitignored) Local .whl packages for offline clusters
├── work/                                           # (Gitignored) Extracted BraTS 2021 volumes# Detecting-Tumor-Region-Hallucinations-Missing-MRI-Modalities-Synthesis
Pipeline for data preparation, multi-modal MRI synthesis, and detecting tumor-region hallucinations across missing BraTS 2021 brain MRI modalities (FLAIR, T1, T1ce, T2) using ground-truth tumor segmentation masks.

├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation & Offline Wheel Setup

### Option A: Offline / Air-Gapped Compute Nodes (Local Wheels)
If running on a restricted HPC cluster or offline Kaggle environment, place pre-downloaded `.whl` files inside `./wheels` and run:

```bash
pip install --no-index --find-links=./wheels \
    numpy pandas scipy scikit-learn matplotlib \
    torch nibabel pydicom transformers datasets \
    sentence-transformers evaluate accelerate nltk
```

### Option B: Standard Online Installation
```bash
git clone https://github.com/<your-username>/Detecting-Tumor-Region-Hallucinations-Missing-MRI-Modalities-Synthesis.git
cd Detecting-Tumor-Region-Hallucinations-Missing-MRI-Modalities-Synthesis

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install numpy pandas scipy scikit-learn matplotlib torch nibabel pydicom tqdm
```

---

## Data Preparation Workflow

1. **Configure Kaggle API Credentials:**
   Place your `kaggle.json` file in `~/.kaggle/kaggle.json` (`chmod 600 ~/.kaggle/kaggle.json`).

2. **Download & Extract BraTS 2021 Data:**
   Run the notebook `01_setup_and_brats2021_data_loading.ipynb` or execute the extraction workflow to unpack the dataset archive into `work/brats2021_task1/`:
   ```bash
   kaggle datasets download -d dschettler8845/brats-2021-task1 -p ./data
   tar -xf ./data/BraTS2021_Training_Data.tar -C ./work/brats2021_task1
   ```

3. **Verify Volume Alignment & Labels:**
   The loader validates shape consistency $(240, 240, 155)$, intensity ranges, and unique segmentation classes (`[0., 1., 2., 4.]`) across all patient folders (`BraTS2021_XXXXX`).

---

## Hallucination Evaluation Framework

To detect hallucinations in synthesized target modalities $\hat{X}_m$ given available source modalities $X_{\setminus m}$:

1. **Region-Stratified Fidelity:**
   Compute image similarity metrics separately across three anatomical masks derived from `*_seg.nii.gz`:
   - **Whole Tumor (WT):** Labels $\{1, 2, 4\}$
   - **Tumor Core (TC):** Labels $\{1, 4\}$
   - **Enhancing Tumor (ET):** Label $\{4\}$
   - **Healthy Tissue (HT):** Label $\{0\}$ (brain mask minus WT)
2. **Downstream Segmentation Consistency:**
   Pass synthesized scans $\hat{X}_m$ into a pre-trained BraTS segmentation network and compare predicted tumor masks against ground-truth masks using Dice Similarity Coefficient ($\text{DSC}$) and 95% Hausdorff Distance ($\text{HD}_{95}$) to flag fabricated or missing tumor regions.

---

## Citation & Acknowledgments

If you use the BraTS 2021 dataset or build upon this pipeline, please cite the official BraTS benchmark papers:

```bibtex
@article{baid2021rsna,
  title={The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification},
  author={Baid, Ujjwal and Ghodasara, Satyam and Mohan, Suyash and Bilello, Michel and Calabrese, Evan and Colak, Errol and Farahani, Keyvan and Kalpathy-Cramer, Jayashree and Kitamura, Felipe C and Pati, Sarthak and others},
  journal={arXiv preprint arXiv:2107.02314},
  year={2021}
}
```

- Dataset packaging reference: `dschettler8845/brats-2021-task1` & RSNA-MICCAI Brain Tumor Radiogenomic Classification.
