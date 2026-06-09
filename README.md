# Context-Aware Attention-Gated Multimodal Fusion (CAAG-MF)

## Overview
This repository contains the implementation of the proposed
Context-Aware Attention-Gated Multimodal Fusion framework for
Emergency Vehicle Detection under adverse environmental conditions.

## Features
- Visual feature extraction
- Acoustic siren analysis
- V2X communication integration
- Attention-gated multimodal fusion
- Real-time emergency vehicle detection

## Installation
Step 1 — Prerequisites
```bash
python3 --version
```
Step 2 — Extract the zip
```bash
unzip caagmf_implementation.zip
cd caagmf
```
Step 3 — Create a virtual environment
```bash
python3 -m venv caagmf_env
caagmf_env\Scripts\activate
```
Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
For GPU support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Step 5 — Run the demo (no data needed)
This verifies the full forward pass works correctly with synthetic data:
```bash
python3 inference.py
```
```bash
Expected output:
```bash
=======================================================
  CA-AGMF — Demo Forward Pass (synthetic data)
=======================================================
  Device: CPU

  Inputs:
    frames:  (2, 3, 224, 224)
    audio:   (2, 1, 168, 128)
    beacon:  (2, 6)
    context: (2, 3)

  Outputs:

  Sample 1 (Night+rain):
    EV detected:  False
    Probability:  0.4877
    Gate weights: g1(vis)=0.3056  g2(aud)=0.3911  g3(v2x)=0.3033

  Sample 2 (Clear day):
    EV detected:  False
    Probability:  0.4893
    Gate weights: g1(vis)=0.2884  g2(aud)=0.3994  g3(v2x)=0.3121

  Parameter counts:
    visual_encoder           :   24,558,144
    audio_encoder            :    1,832,672
    v2x_encoder              :      174,400
    context_encoder          :            0
    attention_gate           :           59
    detection_head           :      131,585
    TOTAL                    :   26,696,860
    gate_%_of_total          :     0.000221

=======================================================
```
Step 6 — Prepare the dataset
```bash
data/
├── train/
│   ├── index.txt
│   ├── frames/        ← .jpg images
│   ├── audio/         ← .npy MFCC+mel arrays (168, T)
│   └── beacons/       ← .npy beacon vectors (6,)
├── val/
│   └── index.txt
└── test/
    └── index.txt
```

## Dataset
Describe the dataset used in the study.

## Model Architecture
![Architecture](docs/architecture.png)
