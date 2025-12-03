# Multimodal Gesture Recognition: LMC + RGB Fusion

This module extends the existing Leap Motion hand-pose detection project with multimodal fusion capabilities, combining **Leap Motion Controller (LMC)** 3D skeletal data with **RGB camera** facial landmarks for enhanced gesture recognition using GRU models.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Training](#training)
- [Inference](#inference)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Examples](#examples)

---

## 🎯 Overview

### Key Features

- **Multimodal Fusion**: Combines manual (hand) and non-manual (facial) features
- **Multiple Encoder Types**: MLP, CNN, LSTM, and Transformer encoders
- **Fusion Strategies**: Concatenation, weighted fusion, gated fusion, cross-modal attention
- **Temporal Modeling**: Bidirectional GRU for sequence processing
- **Real-time Inference**: Simultaneous LMC + RGB capture and prediction
- **Synchronized Data Collection**: Timestamp-based alignment of modalities

### System Requirements

- Python 3.8+
- Leap Motion Controller
- RGB camera (webcam)
- CUDA-capable GPU (recommended)

---

## 🏗️ Architecture

### Pipeline Overview

```
┌──────────────┐       ┌──────────────┐
│ LMC Capture  │       │ RGB Capture  │
│ (27 joints)  │       │ (468 landmarks)│
└──────┬───────┘       └──────┬────────┘
       │                      │
       v                      v
┌──────────────┐       ┌──────────────┐
│ Geometric    │       │ Landmark     │
│ Features     │       │ Features     │
│ (115 dim)    │       │ (189 dim)    │
└──────┬───────┘       └──────┬────────┘
       │                      │
       v                      v
┌──────────────┐       ┌──────────────┐
│ LMC Encoder  │       │ RGB Encoder  │
│ (256 dim)    │       │ (256 dim)    │
└──────┬───────┘       └──────┬────────┘
       │                      │
       └──────────┬───────────┘
                  v
          ┌──────────────┐
          │ Fusion Layer │
          │ (512 dim)    │
          └──────┬───────┘
                 v
          ┌──────────────┐
          │ GRU (BiLSTM) │
          │ (256 hidden) │
          └──────┬───────┘
                 v
          ┌──────────────┐
          │  Classifier  │
          │ (N classes)  │
          └──────────────┘
```

### Feature Extraction

#### LMC Features (115 dimensions)
- **Raw joints**: 27 joints × 3 coordinates = 81 features
- **Distances**: 10 key distances (fingertip-to-wrist, inter-finger)
- **Angles**: 15 joint angles (finger flexion)
- **Palm features**: 9 features (center, radius, normal, orientation)

#### RGB Features (189 dimensions)
- **Facial landmarks**: 62 key landmarks × 3 coordinates = 186 features
- **Head pose**: 3 angles (pitch, yaw, roll)

---

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Leap Motion SDK

Make sure the Leap Motion SDK is installed and the service is running.

```bash
# Install leapc-cffi (already in project)
cd leapc-cffi
pip install -e .

# Install leapc-python-api
cd ../leapc-python-api
pip install -e .
```

### 3. Verify Installation

```bash
# Test LMC connection
python src/lmc/lmc_collector.py --output test_lmc.json --duration 2

# Test RGB camera
python src/rgb/rgb_collector.py --output test_rgb.json --duration 2
```

---

## 📊 Data Collection

### Single Gesture Collection

```bash
python src/data_collection/collect_data.py \
    --gesture "thumbs_up" \
    --output-dir ./data/gestures \
    --samples 10 \
    --duration 5.0 \
    --fps 30
```

### Multiple Gestures Dataset

```bash
python src/data_collection/collect_data.py \
    --gestures "thumbs_up" "peace_sign" "okay" "wave" "fist" \
    --output-dir ./data/gestures \
    --samples 20 \
    --duration 5.0 \
    --fps 30
```

### Data Organization

Collected data will be organized as:

```
data/gestures/
├── thumbs_up/
│   ├── thumbs_up_1234567890_synchronized.json
│   ├── thumbs_up_1234567891_synchronized.json
│   └── ...
├── peace_sign/
│   └── ...
└── collection_log.json
```

### Data Format

Each synchronized JSON file contains:

```json
[
  {
    "timestamp": 1234567890.123,
    "lmc": [x1, y1, z1, ..., x27, y27, z27],
    "lmc_features": [115 features],
    "rgb": [[x1, y1, z1], ..., [x468, y468, z468]],
    "rgb_features": [189 features],
    "pose": {"pitch": 10.5, "yaw": -5.2, "roll": 2.1},
    "time_diff": 0.001
  },
  ...
]
```

---

## 🎓 Training

### Prepare Dataset

Organize your data into train/val splits:

```
data/
├── train/
│   ├── gesture1/
│   │   └── *.json
│   └── gesture2/
│       └── *.json
└── val/
    ├── gesture1/
    └── gesture2/
```

### Basic Training

```bash
python src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001 \
    --save-dir ./checkpoints \
    --log-dir ./logs
```

### Advanced Training Options

```bash
python src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --batch-size 32 \
    --sequence-length 30 \
    --lmc-encoder mlp \
    --rgb-encoder lstm \
    --fusion-type cross_attention \
    --use-cross-attention \
    --gru-hidden-dim 256 \
    --gru-layers 2 \
    --gru-bidirectional \
    --epochs 100 \
    --lr 0.001 \
    --device cuda \
    --save-dir ./checkpoints \
    --log-dir ./logs
```

### Encoder Options

- **LMC Encoder**: `mlp`, `cnn`, `hybrid`
- **RGB Encoder**: `mlp`, `cnn`, `lstm`, `transformer`

### Fusion Options

- **Fusion Type**: `concat`, `weighted`, `gated`, `bilinear`, `cross_attention`, `co_attention`

### Monitor Training

```bash
tensorboard --logdir ./logs
```

---

## 🔮 Inference

### Real-time Inference

```bash
python src/training/inference.py \
    --model ./checkpoints/best_model.pth \
    --labels ./data/label_mapping.json \
    --sequence-length 30 \
    --confidence-threshold 0.5 \
    --camera 0
```

### Save Output Video

```bash
python src/training/inference.py \
    --model ./checkpoints/best_model.pth \
    --labels ./data/label_mapping.json \
    --save-video output_demo.mp4
```

### Create Label Mapping

Before inference, create a label mapping file:

```json
{
  "idx_to_label": {
    "0": "thumbs_up",
    "1": "peace_sign",
    "2": "okay",
    "3": "wave",
    "4": "fist"
  }
}
```

---

## 📁 Project Structure

```
src/
├── data_collection/
│   └── collect_data.py           # Unified data collection
├── lmc/
│   ├── __init__.py
│   └── lmc_collector.py          # LMC data capture
├── rgb/
│   ├── __init__.py
│   └── rgb_collector.py          # RGB camera capture
├── utils/
│   ├── __init__.py
│   ├── geometry_features.py      # LMC geometric features
│   ├── landmark_utils.py         # Facial landmark processing
│   ├── sync.py                   # Stream synchronization
│   └── dataset.py                # PyTorch dataset loader
├── models/
│   ├── encoders/
│   │   ├── __init__.py
│   │   ├── lmc_encoder.py        # LMC feature encoders
│   │   └── rgb_encoder.py        # RGB feature encoders
│   └── gru/
│       ├── __init__.py
│       └── multimodal_gru.py     # Complete GRU model
├── fusion/
│   ├── __init__.py
│   ├── fusion.py                 # Fusion strategies
│   └── attention.py              # Cross-modal attention
└── training/
    ├── __init__.py
    ├── train.py                  # Training script
    └── inference.py              # Real-time inference
```

---

## ⚙️ Configuration

### Model Configuration

Create a `config.yaml` file:

```yaml
# Model architecture
model:
  lmc_input_dim: 115
  rgb_input_dim: 189
  encoder_output_dim: 256
  fusion_output_dim: 512
  
  lmc_encoder:
    type: mlp
    hidden_dims: [256, 256, 128]
    dropout: 0.2
  
  rgb_encoder:
    type: lstm
    hidden_dim: 256
    num_layers: 2
    bidirectional: true
    dropout: 0.2
  
  fusion:
    type: cross_attention
    num_heads: 8
    dropout: 0.1
  
  gru:
    hidden_dim: 256
    num_layers: 2
    bidirectional: true
    dropout: 0.2

# Training
training:
  batch_size: 32
  sequence_length: 30
  epochs: 100
  learning_rate: 0.001
  weight_decay: 1e-5
  
# Data
data:
  train_dir: ./data/train
  val_dir: ./data/val
  augmentation: true
  num_workers: 4
```

---

## 📝 Examples

### Example 1: Quick Start

```bash
# 1. Collect data for 3 gestures
python src/data_collection/collect_data.py \
    --gestures "thumbs_up" "peace" "okay" \
    --output-dir ./data/raw \
    --samples 15

# 2. Split into train/val (manually or with a script)

# 3. Train model
python src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --epochs 50

# 4. Run inference
python src/training/inference.py \
    --model ./checkpoints/best_model.pth \
    --labels ./data/labels.json
```

### Example 2: Advanced Pipeline

```python
import torch
from models.gru.multimodal_gru import create_multimodal_gru_model

# Create model
model = create_multimodal_gru_model(
    num_classes=10,
    lmc_encoder_type='cnn',
    rgb_encoder_type='transformer',
    fusion_type='cross_attention',
    use_cross_attention=True,
    gru_bidirectional=True
)

# Load pretrained weights
checkpoint = torch.load('best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Inference
model.eval()
with torch.no_grad():
    predictions = model.predict(lmc_features, rgb_features)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. LMC Connection Failed**
- Ensure Leap Motion service is running
- Check USB connection
- Try restarting the Leap Motion service

**2. Camera Not Found**
- Check camera permissions
- Try different camera IDs: `--camera 0`, `--camera 1`
- Ensure camera is not being used by another application

**3. MediaPipe Not Working**
- Install with: `pip install mediapipe --upgrade`
- For Linux, may need additional dependencies

**4. CUDA Out of Memory**
- Reduce batch size: `--batch-size 16`
- Reduce sequence length: `--sequence-length 20`
- Use smaller model dimensions

---

## 📚 References

### Key Papers

1. **Multimodal Fusion**: "Multimodal Machine Learning: A Survey and Taxonomy" (Baltrušaitis et al., 2019)
2. **Cross-Modal Attention**: "Attention is All You Need" (Vaswani et al., 2017)
3. **Gesture Recognition**: "Hand Gesture Recognition: A Literature Review" (Rautaray & Agrawal, 2015)

### Related Work

- MediaPipe: https://google.github.io/mediapipe/
- Leap Motion SDK: https://developer.leapmotion.com/
- PyTorch: https://pytorch.org/

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

This project extends the original Leap Motion hand-pose detection system. Please refer to the main project license.

---

## 👥 Authors

- Original LMC implementation: [Previous contributors]
- Multimodal fusion extension: [Your name/team]

---

## 🙏 Acknowledgments

- Leap Motion for the hand tracking SDK
- Google MediaPipe for facial landmark detection
- PyTorch team for the deep learning framework

---

**Happy Gesture Recognition! 🤟**
