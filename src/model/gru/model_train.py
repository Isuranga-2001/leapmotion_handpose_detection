import numpy as np
import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import joblib
import argparse


def load_gru_data(data_dir="data/gru_data/v2"):
    """
    Load GRU sequence data from .npy files.
    
    Args:
        data_dir: Directory containing .npy files with naming format: {class_name}_{timestamp}.npy
    
    Returns:
        X: numpy array of sequences (samples, timesteps, features)
        y: numpy array of class labels
        class_names: list of unique class names
    """
    print(f"[INFO] Loading GRU data from {data_dir}")
    
    # Find all .npy files
    file_pattern = os.path.join(data_dir, "*.npy")
    npy_files = glob.glob(file_pattern)
    
    if not npy_files:
        raise ValueError(f"No .npy files found in {data_dir}")
    
    sequences = []
    labels = []
    
    for file_path in npy_files:
        filename = os.path.basename(file_path)
        class_name = filename.split('_')[0]  # Extract class name before first underscore
        
        try:
            # Load sequence data
            sequence = np.load(file_path)
            
            # Validate sequence shape
            if sequence.ndim != 2:
                print(f"[WARNING] Skipping {filename}: Expected 2D array, got {sequence.ndim}D")
                continue
                
            sequences.append(sequence)
            labels.append(class_name)
            
        except Exception as e:
            print(f"[WARNING] Error loading {filename}: {e}")
            continue
    
    if not sequences:
        raise ValueError("No valid sequence files found")
    
    # Convert to numpy arrays
    X = np.array(sequences, dtype=np.float32)
    y = np.array(labels)
    
    # Get unique class names
    class_names = sorted(np.unique(y))
    
    print(f"[INFO] Loaded {len(sequences)} sequences")
    print(f"[INFO] Sequence shape: {X.shape}")  # (samples, timesteps, features)
    print(f"[INFO] Classes found: {class_names}")
    
    # Class distribution
    for class_name in class_names:
        count = np.sum(y == class_name)
        print(f"  {class_name}: {count} samples ({count/len(y)*100:.1f}%)")
    
    return X, y, class_names


def preprocess_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Preprocess the data for GRU training.
    
    Args:
        X: Input sequences (samples, timesteps, features)
        y: Labels
        test_size: Proportion for test set
        val_size: Proportion of training set used for validation
        random_state: Random seed
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    """
    print(f"[INFO] Preprocessing data...")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Convert to categorical (one-hot encoding)
    num_classes = len(label_encoder.classes_)
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    # Split into train and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y_categorical, test_size=test_size, stratify=y_encoded, random_state=random_state
    )
    
    # Split train into train and validation
    y_train_val_labels = np.argmax(y_train_val, axis=1)  # Convert back to labels for stratification
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val_labels, random_state=random_state
    )
    
    print(f"[INFO] Training set: {X_train.shape[0]} samples")
    print(f"[INFO] Validation set: {X_val.shape[0]} samples")
    print(f"[INFO] Test set: {X_test.shape[0]} samples")
    print(f"[INFO] Number of classes: {num_classes}")
    
    # Feature normalization (optional - normalize across time and features)
    # Calculate statistics from training data only
    train_mean = np.mean(X_train, axis=(0, 1), keepdims=True)
    train_std = np.std(X_train, axis=(0, 1), keepdims=True)
    train_std = np.where(train_std == 0, 1.0, train_std)  # Avoid division by zero
    
    # Apply normalization to all sets
    X_train = (X_train - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    
    print(f"[INFO] Data normalized using training set statistics")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, train_mean, train_std


def build_gru_model(input_shape, num_classes, gru_units=[128, 64], dropout_rate=0.3, learning_rate=0.001):
    """
    Build and compile a GRU model for sequence classification.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        num_classes: Number of output classes
        gru_units: List of GRU layer units
        dropout_rate: Dropout rate for regularization
        learning_rate: Learning rate for optimizer
    
    Returns:
        Compiled Keras model
    """
    print(f"[INFO] Building GRU model...")
    print(f"[INFO] Input shape: {input_shape}")
    print(f"[INFO] Number of classes: {num_classes}")
    print(f"[INFO] GRU units: {gru_units}")
    
    model = Sequential()
    
    # First GRU layer
    model.add(GRU(
        gru_units[0], 
        return_sequences=True if len(gru_units) > 1 else False,
        input_shape=input_shape,
        name='gru_1'
    ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Additional GRU layers
    for i, units in enumerate(gru_units[1:], 2):
        return_seq = i < len(gru_units)  # Return sequences for all but last layer
        model.add(GRU(units, return_sequences=return_seq, name=f'gru_{i}'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    
    # Dense layers for classification
    model.add(Dense(64, activation='relu', name='dense_1'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    model.add(Dense(32, activation='relu', name='dense_2'))
    model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax', name='output'))
    
    # Compile model
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"[INFO] Model compiled successfully")
    model.summary()
    
    return model


def train_gru_model(model, X_train, X_val, y_train, y_val, 
                   epochs=100, batch_size=32, model_save_path="model/gru/gru_leapmotion_gestures.h5"):
    """
    Train the GRU model with callbacks.
    
    Args:
        model: Compiled Keras model
        X_train, X_val, y_train, y_val: Training and validation data
        epochs: Maximum number of epochs
        batch_size: Batch size for training
        model_save_path: Path to save the best model
    
    Returns:
        history: Training history
        best_model: Best model (loaded from checkpoint)
    """
    print(f"[INFO] Starting model training...")
    print(f"[INFO] Epochs: {epochs}, Batch size: {batch_size}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"[INFO] Training completed!")
    print(f"[INFO] Best model saved to: {model_save_path}")
    
    # Load the best model
    best_model = tf.keras.models.load_model(model_save_path)
    
    return history, best_model


def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluate the trained model on test data.
    
    Args:
        model: Trained Keras model
        X_test, y_test: Test data and labels
        label_encoder: Fitted LabelEncoder
    
    Returns:
        test_accuracy: Test accuracy score
    """
    print(f"[INFO] Evaluating model on test set...")
    
    # Make predictions
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Calculate accuracy
    test_accuracy = accuracy_score(y_true, y_pred)
    
    print(f"\n[RESULT] Test Accuracy: {test_accuracy*100:.2f}%\n")
    
    # Detailed classification report
    class_names = label_encoder.classes_
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return test_accuracy


def plot_training_history(history, save_path="model/gru/training_history.png"):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Training history from model.fit()
        save_path: Path to save the plot
    """
    print(f"[INFO] Plotting training history...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training & validation accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training & validation loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[INFO] Training history plot saved to: {save_path}")
    plt.close()


def save_model_artifacts(model, label_encoder, train_mean, train_std, test_accuracy, 
                        save_dir="model/gru"):
    """
    Save all model artifacts including preprocessing parameters.
    
    Args:
        model: Trained Keras model
        label_encoder: Fitted LabelEncoder
        train_mean, train_std: Normalization parameters
        test_accuracy: Test accuracy score
        save_dir: Directory to save artifacts
    """
    print(f"[INFO] Saving model artifacts...")
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save preprocessing parameters and metadata
    artifacts = {
        'label_encoder': label_encoder,
        'train_mean': train_mean,
        'train_std': train_std,
        'test_accuracy': test_accuracy,
        'model_path': os.path.join(save_dir, 'gru_leapmotion_gestures.h5')
    }
    
    artifacts_path = os.path.join(save_dir, 'model_artifacts.pkl')
    joblib.dump(artifacts, artifacts_path)
    
    print(f"[INFO] Model artifacts saved to: {artifacts_path}")
    print(f"[INFO] Artifacts include: label encoder, normalization parameters, and metadata")


def load_model_artifacts(artifacts_path="model_artifacts.pkl"):
    """
    Load saved model artifacts.
    
    Args:
        artifacts_path: Path to saved artifacts
    
    Returns:
        Dictionary containing all model artifacts
    """
    artifacts = joblib.load(artifacts_path)
    
    # Handle model path - check if it's absolute or needs to be made relative to artifacts_path
    model_path = artifacts['model_path']
    
    # If model_path is not absolute and doesn't exist, try to construct it relative to artifacts_path
    if not os.path.isabs(model_path) or not os.path.exists(model_path):
        # Get the directory containing the artifacts file
        artifacts_dir = os.path.dirname(os.path.abspath(artifacts_path))
        # Try to find the model file in the same directory
        model_filename = os.path.basename(model_path)
        alternative_model_path = os.path.join(artifacts_dir, model_filename)
        
        if os.path.exists(alternative_model_path):
            model_path = alternative_model_path
        else:
            # If still not found, try common model filename
            common_model_path = os.path.join(artifacts_dir, 'gru_leapmotion_gestures.h5')
            if os.path.exists(common_model_path):
                model_path = common_model_path
    
    model = tf.keras.models.load_model(model_path, compile=False)
    artifacts['model'] = model
    return artifacts


def predict_gesture(model, label_encoder, train_mean, train_std, sequence):
    """
    Predict gesture from a single sequence.
    
    Args:
        model: Trained Keras model
        label_encoder: Fitted LabelEncoder
        train_mean, train_std: Normalization parameters
        sequence: Input sequence (timesteps, features) or (1, timesteps, features)
    
    Returns:
        predicted_class: Predicted gesture class name
        confidence: Prediction confidence
        class_probabilities: All class probabilities
    """
    # Ensure sequence is 3D: (1, timesteps, features)
    if sequence.ndim == 2:
        sequence = np.expand_dims(sequence, axis=0)
    
    # Apply normalization
    sequence_normalized = (sequence - train_mean) / train_std
    
    # Predict
    probabilities = model.predict(sequence_normalized, verbose=0)[0]
    predicted_idx = np.argmax(probabilities)
    
    # Decode prediction
    predicted_class = label_encoder.inverse_transform([predicted_idx])[0]
    confidence = probabilities[predicted_idx]
    
    # Create class probability dictionary
    class_probabilities = {
        label_encoder.classes_[i]: prob 
        for i, prob in enumerate(probabilities)
    }
    
    return predicted_class, confidence, class_probabilities


def main():
    parser = argparse.ArgumentParser(description="Train GRU model for LeapMotion hand gesture classification")
    parser.add_argument("--data_dir", type=str, default="data/gru_data/v2",
                       help="Directory containing .npy sequence files (default: data/gru_data/v2)")
    parser.add_argument("--model_dir", type=str, default="model/gru",
                       help="Directory to save trained model and artifacts (default: model/gru)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Maximum number of training epochs (default: 100)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    parser.add_argument("--test_size", type=float, default=0.2,
                       help="Test size fraction (default: 0.2)")
    parser.add_argument("--val_size", type=float, default=0.2,
                       help="Validation size fraction of training set (default: 0.2)")
    parser.add_argument("--gru_units", type=int, nargs="+", default=[128, 64],
                       help="GRU layer units (default: 128 64)")
    parser.add_argument("--dropout_rate", type=float, default=0.3,
                       help="Dropout rate (default: 0.3)")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        return
    
    try:
        # Load data
        X, y, class_names = load_gru_data(args.data_dir)
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder, train_mean, train_std = preprocess_data(
            X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.random_state
        )
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
        num_classes = len(class_names)
        
        model = build_gru_model(
            input_shape=input_shape,
            num_classes=num_classes,
            gru_units=args.gru_units,
            dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate
        )
        
        # Train model
        model_save_path = os.path.join(args.model_dir, "gru_leapmotion_gestures.h5")
        history, best_model = train_gru_model(
            model, X_train, X_val, y_train, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=model_save_path
        )
        
        # Evaluate model
        test_accuracy = evaluate_model(best_model, X_test, y_test, label_encoder)
        
        # Plot training history
        plot_history_path = os.path.join(args.model_dir, "training_history.png")
        plot_training_history(history, plot_history_path)
        
        # Save all artifacts
        save_model_artifacts(best_model, label_encoder, train_mean, train_std, test_accuracy, args.model_dir)
        
        print(f"\n[SUCCESS] Training completed!")
        print(f"[INFO] Final test accuracy: {test_accuracy*100:.2f}%")
        print(f"[INFO] All model files saved to: {args.model_dir}")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
