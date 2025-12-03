# Quick Reference Guide

## 🚀 Quick Start Commands

### 1. Setup
```bash
python src/setup_multimodal.py
```

### 2. Collect Data (Single Gesture)
```bash
python src/data_collection/collect_data.py \
    --gesture "thumbs_up" \
    --output-dir ./data/raw \
    --samples 10 \
    --duration 5
```

### 3. Collect Full Dataset
```bash
python src/data_collection/collect_data.py \
    --gestures "thumbs_up" "peace" "okay" "wave" "fist" \
    --output-dir ./data/raw \
    --samples 20 \
    --duration 5
```

### 4. Train Model
```bash
python src/training/train.py \
    --train-dir ./data/train \
    --val-dir ./data/val \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.001
```

### 5. Real-time Inference
```bash
python src/training/inference.py \
    --model ./checkpoints/best_model.pth \
    --labels ./data/labels.json
```

---

## 📊 Model Configurations

### Configuration 1: Fast & Simple (MLP)
```bash
python src/training/train.py \
    --lmc-encoder mlp \
    --rgb-encoder mlp \
    --fusion-type concat \
    --encoder-dim 128 \
    --gru-hidden-dim 128
```
**Use when**: Quick prototyping, limited compute

### Configuration 2: Balanced (CNN + LSTM)
```bash
python src/training/train.py \
    --lmc-encoder cnn \
    --rgb-encoder lstm \
    --fusion-type weighted \
    --encoder-dim 256 \
    --gru-hidden-dim 256 \
    --gru-bidirectional
```
**Use when**: Good balance of performance and speed

### Configuration 3: Maximum Performance (Cross-Attention)
```bash
python src/training/train.py \
    --lmc-encoder hybrid \
    --rgb-encoder transformer \
    --fusion-type cross_attention \
    --use-cross-attention \
    --encoder-dim 256 \
    --fusion-dim 512 \
    --gru-hidden-dim 256 \
    --gru-layers 2 \
    --gru-bidirectional
```
**Use when**: Best accuracy needed, sufficient GPU memory

---

## 🎯 Common Tasks

### View TensorBoard Logs
```bash
tensorboard --logdir ./logs
```

### Resume Training
```bash
python src/training/train.py \
    --resume ./checkpoints/checkpoint_epoch_50.pth \
    --train-dir ./data/train \
    --val-dir ./data/val
```

### Test Single Modality
```bash
# Test LMC only
python src/lmc/lmc_collector.py --output test.json --duration 3

# Test RGB only
python src/rgb/rgb_collector.py --output test.json --duration 3
```

### Check Data Synchronization
```python
from utils.sync import load_and_synchronize, compute_synchronization_stats

# Load and sync
frames = load_and_synchronize('lmc.json', 'rgb.json', target_fps=30)

# Check quality
stats = compute_synchronization_stats(frames)
print(f"Mean time diff: {stats['mean_time_diff']:.4f}s")
```

---

## 🔧 Troubleshooting

### Issue: "CUDA out of memory"
**Solutions**:
```bash
# Reduce batch size
--batch-size 16

# Reduce sequence length
--sequence-length 20

# Use smaller model
--encoder-dim 128 --gru-hidden-dim 128
```

### Issue: "No Leap Motion detected"
**Solutions**:
1. Check Leap Motion service is running
2. Reconnect USB cable
3. Restart Leap Motion service

### Issue: "Camera not found"
**Solutions**:
```bash
# Try different camera ID
--camera 1

# List available cameras (Linux)
v4l2-ctl --list-devices

# List available cameras (Windows)
python -c "import cv2; print([i for i in range(5) if cv2.VideoCapture(i).isOpened()])"
```

### Issue: "Poor synchronization"
**Solutions**:
- Reduce FPS: `--fps 20`
- Increase max time diff in sync.py
- Ensure stable USB/camera connection

---

## 📈 Performance Tuning

### For Better Accuracy
1. Collect more data (50+ samples per gesture)
2. Use cross-attention fusion
3. Increase model capacity (encoder-dim 512)
4. Use bidirectional GRU
5. Train longer (200+ epochs)

### For Faster Inference
1. Use MLP encoders
2. Use concat fusion
3. Reduce encoder dimensions
4. Use unidirectional GRU
5. Reduce sequence length

### For Less Memory
1. Reduce batch size
2. Use smaller models
3. Disable bidirectional GRU
4. Use gradient checkpointing

---

## 📝 Data Format Examples

### LMC Frame
```json
{
  "timestamp": 1234567890.123,
  "hand": [x1, y1, z1, ..., x27, y27, z27],
  "features": [115 geometric features]
}
```

### RGB Frame
```json
{
  "timestamp": 1234567890.124,
  "facial_landmarks": [[x1, y1, z1], ..., [x468, y468, z468]],
  "pose": {"pitch": 10.5, "yaw": -5.2, "roll": 2.1},
  "features": [189 features]
}
```

### Synchronized Frame
```json
{
  "timestamp": 1234567890.123,
  "lmc": [...],
  "lmc_features": [...],
  "rgb": [...],
  "rgb_features": [...],
  "pose": {...},
  "time_diff": 0.001
}
```

---

## 🎨 Visualization

### Plot Training History
```python
import json
import matplotlib.pyplot as plt

with open('./checkpoints/training_history.json', 'r') as f:
    history = json.load(f)

plt.plot([h['accuracy'] for h in history['train']], label='Train')
plt.plot([h['accuracy'] for h in history['val']], label='Val')
plt.legend()
plt.show()
```

### Visualize Predictions
```python
from training.inference import RealtimeInference

inference = RealtimeInference(
    model_path='./checkpoints/best_model.pth',
    label_map_path='./data/labels.json'
)
inference.run(display=True, save_video='output.mp4')
```

---

## 🔍 Model Analysis

### Count Parameters
```python
from models.gru.multimodal_gru import create_multimodal_gru_model

model = create_multimodal_gru_model(num_classes=10)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
```

### Check Model Output
```python
import torch

model.eval()
with torch.no_grad():
    lmc = torch.randn(1, 30, 115)  # (batch, seq, features)
    rgb = torch.randn(1, 30, 189)
    output = model(lmc, rgb)
    print(f"Output shape: {output.shape}")  # Should be (1, num_classes)
```

---

## 🌟 Best Practices

### Data Collection
- ✅ Collect in varied lighting conditions
- ✅ Use different performers
- ✅ Include variations in speed/style
- ✅ Balance class distribution
- ❌ Don't collect all samples in one session

### Training
- ✅ Start with small model, increase if needed
- ✅ Use validation set for hyperparameter tuning
- ✅ Monitor for overfitting
- ✅ Save checkpoints regularly
- ❌ Don't train without validation data

### Inference
- ✅ Warm up model before timing
- ✅ Use confidence thresholds
- ✅ Smooth predictions over frames
- ✅ Handle missing detections gracefully
- ❌ Don't trust low-confidence predictions

---

## 📞 Support

For issues or questions:
1. Check MULTIMODAL_README.md for detailed docs
2. Review this quick reference
3. Check existing issues in repository
4. Create new issue with error logs

---

**Last Updated**: December 2025
