import joblib
import numpy as np
import pandas as pd
import os
import sys
import time

# Add the parent directory to the path to import functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import leap
    from functions.extract_features import extract_features
    LEAP_AVAILABLE = True
except ImportError:
    print("[WARNING] Leap Motion SDK not available. Only CSV-based prediction will work.")
    LEAP_AVAILABLE = False


def load_trained_model(model_path):
    """
    Load the trained Random Forest model and associated metadata.
    
    Args:
        model_path: Path to the saved model (.pkl file)
        
    Returns:
        model: Trained RandomForestClassifier
        label_encoder: LabelEncoder for class names
        feature_columns: List of expected feature column names
        test_accuracy: Model's test accuracy
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model_data = joblib.load(model_path)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict):
            return (
                model_data['model'],
                model_data['label_encoder'],
                model_data['feature_columns'],
                model_data.get('test_accuracy', 'Unknown')
            )
        else:
            # Old format - just the model
            print("[WARNING] Old model format detected. Label encoding may not work properly.")
            return model_data, None, None, 'Unknown'
            
    except Exception as e:
        raise ValueError(f"Error loading model: {e}")


def predict_from_features(model, label_encoder, feature_columns, feature_vector, show_probabilities=True):
    """
    Make a prediction from a feature vector.
    
    Args:
        model: Trained classifier
        label_encoder: LabelEncoder for class names (can be None)
        feature_columns: Expected feature column names (can be None)
        feature_vector: Feature vector or dictionary of features
        show_probabilities: Whether to show class probabilities
        
    Returns:
        predicted_class: Predicted gesture class
        confidence: Confidence score
        probabilities: Class probabilities (if available)
    """
    # Handle feature vector input
    if isinstance(feature_vector, dict):
        if feature_columns is not None:
            # Extract features in the correct order
            feature_vector = np.array([feature_vector.get(col, 0.0) for col in feature_columns])
        else:
            # Use all numeric values from the dictionary
            feature_vector = np.array([v for v in feature_vector.values() if isinstance(v, (int, float))])
    
    # Ensure feature vector is 2D
    if feature_vector.ndim == 1:
        feature_vector = feature_vector.reshape(1, -1)
    
    # Make prediction
    try:
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        confidence = np.max(probabilities)
        
        # Decode prediction if label encoder is available
        if label_encoder is not None:
            predicted_class = label_encoder.inverse_transform([prediction])[0]
        else:
            predicted_class = str(prediction)
        
        if show_probabilities:
            print(f"[RESULT] Predicted class: {predicted_class}")
            print(f"[RESULT] Confidence: {confidence:.3f}")
            
            if label_encoder is not None:
                print("[RESULT] Class probabilities:")
                class_names = label_encoder.classes_
                for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
                    print(f"  {class_name}: {prob:.3f}")
            else:
                print(f"[RESULT] Raw probabilities: {probabilities}")
        
        return predicted_class, confidence, probabilities
        
    except Exception as e:
        raise ValueError(f"Prediction error: {e}")


def predict_from_csv_row(model_path, csv_path, row_index=0):
    """
    Predict gesture from a specific row in a CSV file.
    
    Args:
        model_path: Path to trained model
        csv_path: Path to CSV file with features
        row_index: Index of row to predict (0-based)
    """
    print(f"[INFO] Loading model from {model_path}")
    model, label_encoder, feature_columns, test_acc = load_trained_model(model_path)
    print(f"[INFO] Model test accuracy: {test_acc}")
    
    print(f"[INFO] Loading CSV data from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if row_index >= len(df):
        print(f"[ERROR] Row index {row_index} out of range. CSV has {len(df)} rows.")
        return
    
    # Get the row data
    row_data = df.iloc[row_index]
    print(f"[INFO] Predicting for row {row_index}")
    
    # Show actual class if available
    if 'class_name' in df.columns:
        actual_class = row_data['class_name']
        print(f"[INFO] Actual class: {actual_class}")
    
    # Extract features (exclude metadata columns)
    exclude_cols = ['timestamp', 'hand_type', 'class_name']
    feature_dict = {col: row_data[col] for col in df.columns if col not in exclude_cols}
    
    print(f"[INFO] Feature vector extracted with {len(feature_dict)} features")
    
    # Make prediction
    predicted_class, confidence, probabilities = predict_from_features(
        model, label_encoder, feature_columns, feature_dict
    )
    
    return predicted_class, confidence


def predict_live_leapmotion(model_path, duration=10):
    """
    Predict gestures from live Leap Motion data.
    
    Args:
        model_path: Path to trained model
        duration: How long to collect predictions (seconds)
    """
    if not LEAP_AVAILABLE:
        print("[ERROR] Leap Motion SDK not available for live prediction.")
        return
    
    print(f"[INFO] Loading model from {model_path}")
    model, label_encoder, feature_columns, test_acc = load_trained_model(model_path)
    print(f"[INFO] Model test accuracy: {test_acc}")
    
    class PredictionListener(leap.Listener):
        def __init__(self, model, label_encoder, feature_columns):
            super().__init__()
            self.model = model
            self.label_encoder = label_encoder
            self.feature_columns = feature_columns
            self.prediction_count = 0
            self.last_prediction_time = 0
            self.prediction_interval = 1.0  # Predict every 1 second
    
        def on_connection_event(self, event):
            print("[INFO] Connected to Leap Motion device")
    
        def on_tracking_event(self, event):
            current_time = time.time()
            
            if current_time - self.last_prediction_time < self.prediction_interval:
                return
                
            if event.hands:
                for hand in event.hands:
                    hand_type = "Right" if hand.type == leap.HandType.Right else "Left"
                    
                    # Extract features
                    features = extract_features(hand)
                    
                    print(f"\n{'='*50}")
                    print(f"PREDICTION #{self.prediction_count + 1} - {hand_type} Hand")
                    print(f"{'='*50}")
                    
                    # Make prediction
                    try:
                        predicted_class, confidence, probabilities = predict_from_features(
                            self.model, self.label_encoder, self.feature_columns, features
                        )
                        
                        self.prediction_count += 1
                        self.last_prediction_time = current_time
                        
                    except Exception as e:
                        print(f"[ERROR] Prediction failed: {e}")
            else:
                if current_time - self.last_prediction_time >= self.prediction_interval:
                    print("\n[INFO] No hands detected - place your hand over the sensor")
                    self.last_prediction_time = current_time
    
    listener = PredictionListener(model, label_encoder, feature_columns)
    connection = leap.Connection()
    connection.add_listener(listener)
    
    try:
        with connection.open():
            print(f"\n[INFO] Starting live prediction for {duration} seconds...")
            print("[INFO] Place your hand over the Leap Motion sensor")
            print("[INFO] Press Ctrl+C to stop early")
            
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(0.1)
                
        print(f"\n[INFO] Live prediction completed. Total predictions: {listener.prediction_count}")
        
    except KeyboardInterrupt:
        print(f"\n[INFO] Stopped by user. Total predictions: {listener.prediction_count}")


def test_model_with_dataset(model_path, csv_path, num_samples=10):
    """
    Test the model with multiple samples from a dataset.
    
    Args:
        model_path: Path to trained model
        csv_path: Path to test CSV file
        num_samples: Number of random samples to test
    """
    print(f"[INFO] Loading model from {model_path}")
    model, label_encoder, feature_columns, test_acc = load_trained_model(model_path)
    print(f"[INFO] Model test accuracy: {test_acc}")
    
    print(f"[INFO] Loading test data from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Dataset contains {len(df)} samples")
    
    # Select random samples
    if num_samples > len(df):
        num_samples = len(df)
    
    sample_indices = np.random.choice(len(df), size=num_samples, replace=False)
    
    correct_predictions = 0
    total_predictions = 0
    
    print(f"\n[INFO] Testing with {num_samples} random samples...")
    print("="*80)
    
    for i, idx in enumerate(sample_indices):
        row_data = df.iloc[idx]
        
        # Get actual class if available
        actual_class = row_data.get('class_name', 'Unknown')
        
        # Extract features
        exclude_cols = ['timestamp', 'hand_type', 'class_name']
        feature_dict = {col: row_data[col] for col in df.columns if col not in exclude_cols}
        
        print(f"\nSample {i+1}/{num_samples} (Row {idx}):")
        print(f"Actual class: {actual_class}")
        
        # Make prediction
        try:
            predicted_class, confidence, probabilities = predict_from_features(
                model, label_encoder, feature_columns, feature_dict, show_probabilities=False
            )
            
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.3f}")
            
            # Check if prediction is correct
            if actual_class != 'Unknown':
                is_correct = predicted_class == actual_class
                print(f"Correct: {'✓' if is_correct else '✗'}")
                
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
            
        except Exception as e:
            print(f"Prediction failed: {e}")
    
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print(f"\n{'='*80}")
        print(f"TEST RESULTS:")
        print(f"  Samples tested: {total_predictions}")
        print(f"  Correct predictions: {correct_predictions}")
        print(f"  Test accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Model training accuracy: {test_acc}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Random Forest model for LeapMotion gesture classification")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to trained model (.pkl file)")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # CSV prediction command
    csv_parser = subparsers.add_parser('csv', help='Predict from CSV file')
    csv_parser.add_argument("--csv", type=str, required=True, 
                           help="Path to CSV file with features")
    csv_parser.add_argument("--row", type=int, default=0, 
                           help="Row index to predict (default: 0)")
    
    # Live prediction command
    live_parser = subparsers.add_parser('live', help='Live prediction from Leap Motion')
    live_parser.add_argument("--duration", type=int, default=10, 
                            help="Prediction duration in seconds (default: 10)")
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model with dataset')
    test_parser.add_argument("--csv", type=str, required=True, 
                            help="Path to test CSV file")
    test_parser.add_argument("--samples", type=int, default=10, 
                            help="Number of samples to test (default: 10)")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        exit(1)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        exit(1)
    
    try:
        if args.command == 'csv':
            if not os.path.exists(args.csv):
                print(f"[ERROR] CSV file not found: {args.csv}")
                exit(1)
            predict_from_csv_row(args.model, args.csv, args.row)
            
        elif args.command == 'live':
            predict_live_leapmotion(args.model, args.duration)
            
        elif args.command == 'test':
            if not os.path.exists(args.csv):
                print(f"[ERROR] CSV file not found: {args.csv}")
                exit(1)
            test_model_with_dataset(args.model, args.csv, args.samples)
            
    except Exception as e:
        print(f"[ERROR] {e}")
        exit(1)
