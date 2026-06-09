# Adaptive Context-Aware Multimodal Fusion for Robust Emergency Vehicle Detection in Smart Intersections

## Overview
This repository contains the implementation of the proposed
Adaptive Context-Aware Multimodal Fusion for Robust Emergency Vehicle Detection in Smart Intersections.

## Features
- Visual feature extraction
- Acoustic siren analysis
- V2X communication integration
- Attention-gated multimodal fusion
- Real-time emergency vehicle detection

## Installation
Step 1: Prerequisites
```bash
python3 --version
```
Step 2: Extract the zip
```bash
unzip caagmf_implementation.zip
cd caagmf
```
Step 3: Create a virtual environment
```bash
python3 -m venv caagmf_env
caagmf_env\Scripts\activate
```
Step 4: Install dependencies
```bash
pip install -r requirements.txt
```
For GPU support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Step 5: Run the demo (no data needed)
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
Step 6: Prepare the dataset
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
Each index.txt line follows this format:
```bash
frames/ev_001.jpg  audio/ev_001.npy  beacons/ev_001.npy  1
frames/bg_001.jpg  audio/bg_001.npy  beacons/bg_001.npy  0
```
Preparing audio .npy files from raw audio:
```bash
import librosa
import numpy as np

y, sr = librosa.load("siren.wav", sr=22050)
mel   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
mfcc  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
stack = np.vstack([mfcc, librosa.power_to_db(mel)])  # (168, T)
np.save("audio/ev_001.npy", stack.astype(np.float32))
```
Preparing beacon .npy files:
```bash
import numpy as np

# [v_type, v_speed, v_heading, priority_flag, distance, latency]
# All values min-max normalised to [0, 1]
beacon = np.array([1.0, 0.72, 0.45, 1.0, 0.38, 0.12], dtype=np.float32)
np.save("beacons/ev_001.npy", beacon)
```
Step 7: Train the model
```bash
from train import train

results = train({
    "data_root":  "./data",
    "epochs":     100,
    "batch_size": 32,
    "device":     "cuda",   # or "cpu"
    "lr":         1e-4,
    "patience":   15,
})
```
Or run directly from terminal:
```bash
python3 train.py
```
Training output per epoch:
```bash
Epoch 001/100 | Train loss: 0.4821  F1: 0.8134 | Val loss: 0.4102  F1: 0.8467 | Gate: [0.48, 0.33, 0.19] | 42.3s
Epoch 002/100 | Train loss: 0.3614  F1: 0.8892 | Val loss: 0.3201  F1: 0.9014 | Gate: [0.51, 0.30, 0.19] | 41.8s
  ✓ New best val F1: 0.9014 — checkpoint saved
...
```
```bash
Checkpoints are saved to ./checkpoints/best_caagmf.pth
```
Step 8: Evaluate on adverse conditions
```bash
from train import train
from dataset import build_dataloaders
from losses import WeightedBCELoss
from model import CAAGMF
import torch

device = "cuda"
model  = CAAGMF(pretrained_visual=False).to(device)
ckpt   = torch.load("checkpoints/best_caagmf.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])

loaders   = build_dataloaders("./data")
criterion = WeightedBCELoss()

conditions = ["clear", "nighttime", "rain", "fog", "high_noise"]
for cond in conditions:
    from train import evaluate
    m = evaluate(model, loaders[f"test_{cond}"], criterion, device)
    print(f"{cond:<16} F1: {m['f1']:.4f}  Acc: {m['accuracy']:.4f}  "
          f"Gate: {m['gate_weights']}")
```
Step 9: Run inference on a single sample
```bash
import torch
from inference import CAAgmfInference

engine = CAAgmfInference(
    checkpoint_path = "checkpoints/best_caagmf.pth",
    device          = "cuda",
    threshold       = 0.5,
)

# Load your sample
frame   = torch.randn(1, 3, 640, 640)   # replace with real frame
audio   = torch.randn(1, 1, 168, 128)   # replace with real audio
beacon  = torch.rand(1, 6)              # replace with real beacon
context = torch.tensor([[0.15, 0.80, 0.30]])  # [luminance, weather, SNR]

result = engine.predict(frame, audio, beacon, context)
print(result)
# {
#   "ev_detected":  True,
#   "probability":  0.9412,
#   "latency_ms":   112.3,
#   "gate_weights": {"g1_visual": 0.18, "g2_audio": 0.48, "g3_v2x": 0.34}
# }
```
Step 10: Run the ablation study
```bash
from inference import AblationStudy
from dataset import build_dataloaders

loaders = build_dataloaders("./data")
ablation = AblationStudy(
    base_config = {"feature_dim": 512, "alpha": 3.0},
    device      = "cuda",
)
results = ablation.run(
    test_loader     = loaders["test_clear"],
    checkpoint_path = "checkpoints/best_caagmf.pth",
)
```
## Dataset
```bash
Visual Modality
BDD100K — Berkeley DeepDrive - https://bdd-data.berkeley.edu/

```
