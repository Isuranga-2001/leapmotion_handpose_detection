#!/usr/bin/env python3
"""
Test script for LeapMotion gesture classification model.
This script tests the trained model using the test_data.csv file.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add parent directories to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

def load_trained_model(model_path):
    """
    Load the trained Random Forest model and associated metadata.
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

def test_model_with_test_data(model_path, test_csv_path):
    """
    Test the model with the test_data.csv file and calculate accuracy.
    """
    print(f"[INFO] Loading model from {model_path}")
    model, label_encoder, feature_columns, train_acc = load_trained_model(model_path)
    print(f"[INFO] Model training accuracy: {train_acc}")
    
    print(f"[INFO] Loading test data from {test_csv_path}")
    df = pd.read_csv(test_csv_path)
    print(f"[INFO] Test dataset contains {len(df)} samples")
    
    # Extract features and labels
    exclude_cols = ['timestamp', 'hand_type', 'class_name']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X_test = df[feature_cols].values
    y_true = df['class_name'].values
    
    print(f"[INFO] Using {len(feature_cols)} features for prediction")
    print(f"[INFO] Classes in test data: {np.unique(y_true)}")
    
    # Make predictions
    print("\n[INFO] Making predictions...")
    predictions = []
    confidences = []
    
    for i in range(len(X_test)):
        feature_vector = X_test[i].reshape(1, -1)
        pred = model.predict(feature_vector)[0]
        prob = model.predict_proba(feature_vector)[0]
        confidence = np.max(prob)
        
        # Decode prediction if label encoder is available
        if label_encoder is not None:
            predicted_class = label_encoder.inverse_transform([pred])[0]
        else:
            predicted_class = str(pred)
        
        predictions.append(predicted_class)
        confidences.append(confidence)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, predictions)
    
    print(f"\n{'='*60}")
    print("TEST RESULTS:")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"Training Accuracy: {train_acc}")
    print(f"Total Samples Tested: {len(predictions)}")
    print(f"Average Confidence: {np.mean(confidences):.3f}")
    
    # Detailed classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, predictions))
    
    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, predictions)
    print(cm)
    
    # Show some example predictions
    print(f"\n{'='*60}")
    print("SAMPLE PREDICTIONS:")
    print(f"{'='*60}")
    
    for i in range(min(10, len(df))):
        actual = y_true[i]
        predicted = predictions[i]
        confidence = confidences[i]
        correct = "✓" if actual == predicted else "✗"
        
        print(f"Sample {i+1:2d}: Actual={actual}, Predicted={predicted}, Confidence={confidence:.3f} {correct}")
    
    return accuracy, predictions, confidences

def main():
    """
    Test the trained model with test_data.csv
    """
    print("LeapMotion Gesture Classification - Model Testing")
    print("=" * 60)
    
    # Paths - using relative paths from the model directory
    model_path = "model/rf/rf_leapmotion_gestures.pkl"
    test_data_path = "data/rf_data/test_data.csv"
    
    # Check if files exist
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        print("Please train the model first using model_train.py")
        return
    
    if not os.path.exists(test_data_path):
        print(f"[ERROR] Test data file not found: {test_data_path}")
        print("Please ensure the test_data.csv file exists in the data directory")
        return
    
    try:
        # Test the model with the test data
        print(f"\nTesting model with test_data.csv...")
        print("-" * 60)
        
        accuracy, predictions, confidences = test_model_with_test_data(model_path, test_data_path)
        
        # Additional analysis
        print(f"\n{'='*60}")
        print("ADDITIONAL ANALYSIS:")
        print(f"{'='*60}")
        
        # Load test data for analysis
        import pandas as pd
        df = pd.read_csv(test_data_path)
        
        # Analyze by class
        print("\nPrediction accuracy by class:")
        for class_name in df['class_name'].unique():
            class_mask = df['class_name'] == class_name
            class_predictions = [predictions[i] for i in range(len(predictions)) if class_mask.iloc[i]]
            class_actual = [df['class_name'].iloc[i] for i in range(len(df)) if class_mask.iloc[i]]
            
            if class_predictions:
                class_accuracy = sum(1 for p, a in zip(class_predictions, class_actual) if p == a) / len(class_predictions)
                print(f"  Class {class_name}: {class_accuracy:.3f} ({class_accuracy*100:.1f}%) - {len(class_predictions)} samples")
        
        # Show confidence distribution
        print(f"\nConfidence Statistics:")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
        print(f"  Std deviation: {np.std(confidences):.3f}")
        
        print(f"\n{'='*60}")
        print("Testing completed successfully!")
        print(f"Overall Test Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()