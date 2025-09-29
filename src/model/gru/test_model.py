#!/usr/bin/env python3
"""
Test script to validate the trained GRU model.
This script runs various tests to ensure the model works correctly.
"""

import numpy as np
import os
import sys

# Add current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from model_train import load_model_artifacts, predict_gesture, load_gru_data
except ImportError:
    # If running from a different directory, try to import from the same directory
    import importlib.util
    model_train_path = os.path.join(current_dir, 'model_train.py')
    if os.path.exists(model_train_path):
        spec = importlib.util.spec_from_file_location("model_train", model_train_path)
        model_train = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_train)
        load_model_artifacts = model_train.load_model_artifacts
        predict_gesture = model_train.predict_gesture
        load_gru_data = model_train.load_gru_data
    else:
        raise ImportError("Could not find model_train.py. Make sure it's in the same directory as test_model.py")


def test_model_loading(model_dir="."):
    """Test if the model can be loaded correctly."""
    print("[TEST] Testing model loading...")
    
    try:
        artifacts_path = os.path.join(model_dir, "model_artifacts.pkl")
        if not os.path.exists(artifacts_path):
            print(f"  ❌ FAIL: Model artifacts not found at {artifacts_path}")
            return False
        
        artifacts = load_model_artifacts(artifacts_path)
        
        # Check required components
        required_keys = ['model', 'label_encoder', 'train_mean', 'train_std', 'test_accuracy']
        for key in required_keys:
            if key not in artifacts:
                print(f"  ❌ FAIL: Missing artifact key: {key}")
                return False
        
        print(f"  ✅ PASS: Model loaded successfully")
        print(f"  📊 Test accuracy: {artifacts['test_accuracy']*100:.2f}%")
        print(f"  🏷️  Classes: {list(artifacts['label_encoder'].classes_)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error loading model: {e}")
        return False


def test_prediction_single_sequence(model_dir=".", data_dir="data/gru_data"):
    """Test prediction on a single sequence."""
    print("[TEST] Testing single sequence prediction...")
    
    try:
        # Load model
        artifacts = load_model_artifacts(os.path.join(model_dir, "model_artifacts.pkl"))
        
        # Find a test sequence
        import glob
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        if not npy_files:
            print(f"  ❌ FAIL: No test data found in {data_dir}")
            return False
        
        test_file = npy_files[0]
        sequence = np.load(test_file)
        
        # Make prediction
        predicted_class, confidence, class_probabilities = predict_gesture(
            model=artifacts['model'],
            label_encoder=artifacts['label_encoder'],
            train_mean=artifacts['train_mean'],
            train_std=artifacts['train_std'],
            sequence=sequence
        )
        
        # Validate results
        if predicted_class is None or confidence is None:
            print(f"  ❌ FAIL: Prediction returned None")
            return False
        
        if not isinstance(confidence, (float, np.floating)):
            print(f"  ❌ FAIL: Confidence is not a float: {type(confidence)}")
            return False
        
        if not (0 <= confidence <= 1):
            print(f"  ❌ FAIL: Confidence out of range: {confidence}")
            return False
        
        if predicted_class not in artifacts['label_encoder'].classes_:
            print(f"  ❌ FAIL: Predicted class not in known classes: {predicted_class}")
            return False
        
        filename = os.path.basename(test_file)
        expected_class = filename.split('_')[0]
        
        print(f"  ✅ PASS: Prediction successful")
        print(f"  📁 File: {filename}")
        print(f"  🎯 Expected: {expected_class}")
        print(f"  🤖 Predicted: {predicted_class} (confidence: {confidence:.3f})")
        print(f"  ✓ Correct: {'Yes' if predicted_class == expected_class else 'No'}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error in prediction: {e}")
        return False


def test_prediction_batch(model_dir=".", data_dir="data/gru_data"):
    """Test batch prediction on multiple sequences."""
    print("[TEST] Testing batch prediction...")
    
    try:
        # Load model
        artifacts = load_model_artifacts(os.path.join(model_dir, "model_artifacts.pkl"))
        
        # Load all test data
        import glob
        npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
        
        if len(npy_files) < 2:
            print(f"  ❌ FAIL: Need at least 2 test files, found {len(npy_files)}")
            return False
        
        correct_predictions = 0
        total_predictions = 0
        
        for file_path in npy_files[:5]:  # Test first 5 files
            filename = os.path.basename(file_path)
            expected_class = filename.split('_')[0]
            
            sequence = np.load(file_path)
            predicted_class, confidence, _ = predict_gesture(
                model=artifacts['model'],
                label_encoder=artifacts['label_encoder'],
                train_mean=artifacts['train_mean'],
                train_std=artifacts['train_std'],
                sequence=sequence
            )
            
            if predicted_class == expected_class:
                correct_predictions += 1
            
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        
        print(f"  ✅ PASS: Batch prediction completed")
        print(f"  📊 Accuracy: {accuracy*100:.1f}% ({correct_predictions}/{total_predictions})")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error in batch prediction: {e}")
        return False


def test_model_shapes(model_dir=".", data_dir="data/gru_data"):
    """Test if model input/output shapes are correct."""
    print("[TEST] Testing model input/output shapes...")
    
    try:
        # Load model and data
        artifacts = load_model_artifacts(os.path.join(model_dir, "model_artifacts.pkl"))
        X, y, class_names = load_gru_data(data_dir)
        
        model = artifacts['model']
        
        # Check input shape
        expected_input_shape = X.shape[1:]  # (timesteps, features)
        model_input_shape = model.input_shape[1:]  # Remove batch dimension
        
        if expected_input_shape != model_input_shape:
            print(f"  ❌ FAIL: Input shape mismatch")
            print(f"    Expected: {expected_input_shape}")
            print(f"    Model: {model_input_shape}")
            return False
        
        # Check output shape
        expected_output_classes = len(class_names)
        model_output_classes = model.output_shape[1]
        
        if expected_output_classes != model_output_classes:
            print(f"  ❌ FAIL: Output shape mismatch")
            print(f"    Expected classes: {expected_output_classes}")
            print(f"    Model output: {model_output_classes}")
            return False
        
        print(f"  ✅ PASS: Model shapes are correct")
        print(f"  📏 Input shape: {model_input_shape}")
        print(f"  📤 Output classes: {model_output_classes}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error checking model shapes: {e}")
        return False


def test_normalization(model_dir=".", data_dir="data/gru_data"):
    """Test if normalization parameters are reasonable."""
    print("[TEST] Testing normalization parameters...")
    
    try:
        # Load model artifacts
        artifacts = load_model_artifacts(os.path.join(model_dir, "model_artifacts.pkl"))
        
        train_mean = artifacts['train_mean']
        train_std = artifacts['train_std']
        
        # Check shapes
        X, _, _ = load_gru_data(data_dir)
        expected_shape = (1, 1, X.shape[2])  # (1, 1, features)
        
        if train_mean.shape != expected_shape:
            print(f"  ❌ FAIL: Mean shape mismatch: {train_mean.shape} != {expected_shape}")
            return False
        
        if train_std.shape != expected_shape:
            print(f"  ❌ FAIL: Std shape mismatch: {train_std.shape} != {expected_shape}")
            return False
        
        # Check for reasonable values
        if np.any(train_std <= 0):
            print(f"  ❌ FAIL: Standard deviation contains non-positive values")
            return False
        
        if np.any(np.isnan(train_mean)) or np.any(np.isnan(train_std)):
            print(f"  ❌ FAIL: Normalization parameters contain NaN values")
            return False
        
        if np.any(np.isinf(train_mean)) or np.any(np.isinf(train_std)):
            print(f"  ❌ FAIL: Normalization parameters contain infinite values")
            return False
        
        print(f"  ✅ PASS: Normalization parameters are valid")
        print(f"  📊 Mean range: [{np.min(train_mean):.3f}, {np.max(train_mean):.3f}]")
        print(f"  📊 Std range: [{np.min(train_std):.3f}, {np.max(train_std):.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ❌ FAIL: Error checking normalization: {e}")
        return False


def run_all_tests(model_dir=".", data_dir="data/gru_data"):
    """Run all tests and return overall result."""
    print("="*60)
    print("🧪 GRU Model Validation Tests")
    print("="*60)
    
    tests = [
        ("Model Loading", lambda: test_model_loading(model_dir)),
        ("Model Shapes", lambda: test_model_shapes(model_dir, data_dir)),
        ("Normalization", lambda: test_normalization(model_dir, data_dir)),
        ("Single Prediction", lambda: test_prediction_single_sequence(model_dir, data_dir)),
        ("Batch Prediction", lambda: test_prediction_batch(model_dir, data_dir)),
    ]
    
    results = []
    for test_name, test_func in tests:
        print()
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "="*60)
    print("📋 Test Results Summary")
    print("="*60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The model is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the model and training process.")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the trained GRU model")
    parser.add_argument("--model_dir", type=str, default=".",
                       help="Directory containing trained model (default: current directory)")
    parser.add_argument("--data_dir", type=str, default="../../data/gru_data",
                       help="Directory containing test data (default: ../../data/gru_data)")
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.model_dir):
        print(f"[ERROR] Model directory not found: {args.model_dir}")
        return False
    
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory not found: {args.data_dir}")
        return False
    
    # Run tests
    success = run_all_tests(args.model_dir, args.data_dir)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)