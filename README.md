# Explainable Multimodal Foundation Model for Early Breast Cancer Diagnosis

**Project title:** Explainable Multimodal Foundation Model for Early Breast Cancer Diagnosis through Cross-Modal Fusion of Mammography, Ultrasound, Clinical Metadata, and Vision Foundation Models.

**Batch:** AIML-A3  
**Project guide:** Dr. N. Balayesu  
**Team:** Dalveer Singh, Chakrala Venkat, Chinthala Lavanya, Ettela Maneesh

## Overview

This project investigates an AI-assisted approach to breast-cancer classification by combining mammography, breast ultrasound, and structured clinical information. The proposed research direction is an Adaptive Cross-Modal Foundation Transformer (ACMFT), which is intended to encode the modalities separately, align their representations through cross-modal attention, and produce a malignancy prediction with supporting explanations and an uncertainty estimate.

This repository is under active development. The implemented components and proposed components are distinguished below.

## Current implementation

The notebooks currently shared by the team cover:

- Exploratory data analysis of the VinDr-Mammo dataset.
- Mammography data preparation and patient-grouped train/validation splitting.
- A mammography classification baseline called `MammoNet`.
- A pretrained ConvNeXtV2 image backbone with embeddings for mammography laterality, view, and breast density.
- A three-class prediction task: normal, benign, and malignant.
- Training utilities including class-weighted loss, AdamW, layer-wise learning-rate decay, mixed precision, exponential moving average (EMA), and validation metrics.

The ConvNeXtV2 notebook is configured with `MODALITY = "mammography"`. Its data-loading function contains an ultrasound branch, but that alone does not mean an ultrasound model has been trained.

## Reported mammography validation result

The uploaded ConvNeXtV2 notebook records the following final validation metrics:

| Metric | Value |
|---|---:|
| Accuracy | 0.7526 |
| Balanced accuracy | 0.7569 |
| Macro AUROC | 0.9038 |
| Macro AUPRC | 0.8323 |

These are notebook-reported validation results, not independent external-test results. Reproduce the experiment with the documented data and environment before treating the numbers as independently verified.

## Proposed architecture

The registration abstract proposes an Adaptive Cross-Modal Foundation Transformer (ACMFT) with:

1. A pretrained vision encoder for mammography.
2. A pretrained vision encoder for ultrasound.
3. A dedicated encoder for structured clinical metadata.
4. Cross-modal attention to align and fuse the modality representations.
5. A multi-task prediction head for benign-versus-malignant classification.
6. Visual explanations (including Grad-CAM / attention-based explanations) and calibrated uncertainty.

These are project goals; they should only be marked as implemented after the corresponding code and experiments are added.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── docs/
│   └── current_implementation.md
└── notebooks/
    ├── vindr-eda.ipynb
    └── oct3-convnextv2-mammo-bench-512-tfl(1).ipynb
```

Keep the team's existing notebooks and repository structure if they are already present. Do not overwrite teammates' work.

## Environment setup

The current mammography notebook uses Python with PyTorch and the libraries listed in `requirements.txt`.

Create and activate a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the machine's CUDA driver using the official PyTorch installation selector. The notebook can run on CPU, but training a large ConvNeXtV2 model at 512-pixel resolution is likely to be slow.

## Dataset setup

The uploaded training notebook was configured for a Kaggle dataset path:

```python
DATA_ROOT = "/kaggle/input/datasets/dhruvkarani22/mammobench-v2"
```

Change `DATA_ROOT` to the actual local dataset directory when running outside Kaggle. See [`data/README.md`](data/README.md) for data-handling notes.

Do not commit patient-level medical images, clinical records, private dataset credentials, or large downloaded archives. Follow each dataset's licence and access terms.

## Running the notebooks

Start Jupyter:

```bash
jupyter notebook
```

Open the notebooks in the `notebooks/` directory. Update dataset paths and confirm that the expected CSV and image files are available before running cells.

## Development roadmap

- [x] VinDr-Mammo exploratory data analysis notebook shared by the team.
- [x] Mammography ConvNeXtV2 baseline notebook shared by the team.
- [ ] Reproducible local configuration and preprocessing documentation.
- [ ] Ultrasound dataset exploration and baseline model.
- [ ] Dedicated clinical metadata encoder.
- [ ] Cross-modal attention / ACMFT fusion.
- [ ] Explainability and uncertainty evaluation.
- [ ] Ablation studies and external-cohort evaluation.

## Evaluation

The current mammography notebook reports accuracy, balanced accuracy, class-wise accuracy, macro AUROC, macro AUPRC, and a confusion matrix. Future multimodal experiments should use patient-level splits and report appropriate per-class and aggregate metrics. Comparisons should use consistent data splits and clearly identify validation versus independent test results.

## Medical-use disclaimer

This is an academic research project and is not a medical device. Its outputs must not be used as a substitute for assessment by qualified healthcare professionals.
