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

## Dataset
Describe the dataset used in the study.

## Model Architecture
![Architecture](docs/architecture.png)
