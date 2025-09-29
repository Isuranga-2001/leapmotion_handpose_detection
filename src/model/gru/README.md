# GRU Model for LeapMotion Hand Gesture Recognition

This directory contains a GRU (Gated Recurrent Unit) model implementation for classifying hand gestures from LeapMotion sequence data.

## Overview

The GRU model is designed to classify sequential hand gesture data captured from LeapMotion sensors. Each sequence consists of multiple timesteps with feature vectors containing:

- Palm position (x, y, z) relative to wrist
- Palm normal orientation (x, y, z)
- Grab and pinch strengths
- Finger tip positions (x, y, z) for all 5 fingers, normalized by reference distance
- Timestamp and hand type information

## Files

- `model_train.py` - Main training script for the GRU model
- `predict.py` - Prediction script for trained models
- `requirements.txt` - Required Python packages
- `README.md` - This documentation file

## Installation

1.Install the required dependencies:

```bash
pip install -r requirements.txt
```

2.Make sure you have the LeapMotion data in the correct format (`.npy` files with sequences).

## Data Format

The training data should be organized as `.npy` files with the naming convention:

```file
{class_name}_{timestamp}.npy
```

Each `.npy` file should contain a numpy array of shape `(timesteps, features)` where:

- `timesteps`: Number of frames in the sequence (typically 50)
- `features`: Number of features per frame (typically 25)

Example: `Come_1759126630.npy`, `More_1759126764.npy`

## Training

### Basic Training

To train the model with default parameters:

```bash
python model_train.py
```

### Advanced Training Options

```bash
python model_train.py \
    --data_dir "../../data/gru_data" \
    --model_dir "." \
    --epochs 150 \
    --batch_size 16 \
    --gru_units 256 128 64 \
    --dropout_rate 0.4 \
    --learning_rate 0.001
```

### Training Parameters

- `--data_dir`: Directory containing `.npy` sequence files (default: `data/gru_data`)
- `--model_dir`: Directory to save trained model (default: `model/gru`)
- `--epochs`: Maximum training epochs (default: 100)
- `--batch_size`: Training batch size (default: 32)
- `--test_size`: Test set proportion (default: 0.2)
- `--val_size`: Validation set proportion (default: 0.2)
- `--gru_units`: GRU layer sizes (default: [128, 64])
- `--dropout_rate`: Dropout rate for regularization (default: 0.3)
- `--learning_rate`: Learning rate (default: 0.001)
- `--random_state`: Random seed (default: 42)

## Model Architecture

The GRU model consists of:

1. **Input Layer**: Accepts sequences of shape `(timesteps, features)`
2. **GRU Layers**: One or more GRU layers with batch normalization and dropout
3. **Dense Layers**: Fully connected layers with ReLU activation
4. **Output Layer**: Softmax layer for multi-class classification

### Default Architecture

- GRU Layer 1: 128 units with return_sequences=True
- BatchNormalization + Dropout (0.3)
- GRU Layer 2: 64 units with return_sequences=False
- BatchNormalization + Dropout (0.3)
- Dense Layer 1: 64 units (ReLU)
- BatchNormalization + Dropout (0.3)
- Dense Layer 2: 32 units (ReLU)
- Dropout (0.3)
- Output Layer: softmax (number of classes)

## Training Features

- **Early Stopping**: Prevents overfitting by monitoring validation loss
- **Model Checkpointing**: Saves the best model based on validation accuracy
- **Learning Rate Scheduling**: Reduces learning rate when validation loss plateaus
- **Data Normalization**: Normalizes features using training set statistics
- **Stratified Splitting**: Maintains class balance in train/validation/test splits

## Prediction

### Single File Prediction

```bash
python predict.py file path/to/sequence.npy
```

### Batch Prediction

```bash
python predict.py batch path/to/data/directory --output results.json
```

### Interactive Mode

```bash
python predict.py interactive
```

## Model Artifacts

After training, the following files are saved:

1. `gru_leapmotion_gestures.h5` - Trained Keras model
2. `model_artifacts.pkl` - Preprocessing parameters and metadata:
   - Label encoder for class names
   - Normalization parameters (mean and std)
   - Test accuracy score
3. `training_history.png` - Training curves (loss and accuracy)

## Example Usage

### Training Example

```python
from model_train import load_gru_data, preprocess_data, build_gru_model, train_gru_model

# Load data
X, y, class_names = load_gru_data("data/gru_data")

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, train_mean, train_std = preprocess_data(X, y)

# Build and train model
model = build_gru_model(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=len(class_names))
history, best_model = train_gru_model(model, X_train, X_val, y_train, y_val)
```

### Prediction Example

```python
from model_train import load_model_artifacts, predict_gesture
import numpy as np

# Load trained model
artifacts = load_model_artifacts("model_artifacts.pkl")

# Load a sequence
sequence = np.load("test_sequence.npy")

# Make prediction
predicted_class, confidence, probabilities = predict_gesture(
    model=artifacts['model'],
    label_encoder=artifacts['label_encoder'],
    train_mean=artifacts['train_mean'],
    train_std=artifacts['train_std'],
    sequence=sequence
)

print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
```

## Performance Tips

1. **GPU Acceleration**: Install `tensorflow-gpu` for faster training
2. **Batch Size**: Increase batch size if you have sufficient GPU memory
3. **Sequence Length**: Ensure all sequences have consistent length
4. **Data Augmentation**: Consider data augmentation techniques for small datasets
5. **Hyperparameter Tuning**: Experiment with different GRU units, learning rates, and dropout rates

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or GRU units
2. **Overfitting**: Increase dropout rate, reduce model complexity, or add more data
3. **Poor Performance**: Check data quality, increase model complexity, or adjust learning rate
4. **Import Errors**: Ensure all dependencies are installed correctly

### Data Issues

- **Inconsistent Sequence Lengths**: All sequences should have the same number of timesteps
- **Missing Features**: Ensure all feature dimensions are consistent
- **Class Imbalance**: Consider using class weights or data augmentation

## Model Evaluation

The training script provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Classification Report**: Precision, recall, and F1-score per class
- **Confusion Matrix**: Detailed confusion matrix
- **Training Curves**: Loss and accuracy plots over epochs

## Integration with LeapMotion

This model is designed to work with the LeapMotion hand tracking system. The feature extraction is based on the `extract_features()` function in `functions/extract_features.py`, which processes raw LeapMotion hand data into normalized feature vectors suitable for the GRU model.

For real-time gesture recognition, integrate this model with the LeapMotion listener classes to classify gestures as they are captured.
