import numpy as np
import os
import sys
import tensorflow as tf
import joblib

# Add current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from model_train import load_model_artifacts, predict_gesture
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
    else:
        raise ImportError("Could not find model_train.py. Make sure it's in the same directory as predict.py")


def load_trained_model(model_dir="model/gru"):
    """
    Load the trained GRU model and all associated artifacts.
    
    Args:
        model_dir: Directory containing the trained model
    
    Returns:
        Dictionary containing model and preprocessing artifacts
    """
    artifacts_path = os.path.join(model_dir, "model_artifacts.pkl")
    
    if not os.path.exists(artifacts_path):
        raise FileNotFoundError(f"Model artifacts not found at: {artifacts_path}")
    
    print(f"[INFO] Loading trained model from {model_dir}")
    artifacts = load_model_artifacts(artifacts_path)
    
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Test accuracy: {artifacts['test_accuracy']*100:.2f}%")
    print(f"[INFO] Available classes: {list(artifacts['label_encoder'].classes_)}")
    
    return artifacts


def predict_from_file(model_artifacts, sequence_file):
    """
    Predict gesture from a .npy sequence file.
    
    Args:
        model_artifacts: Dictionary containing model and preprocessing artifacts
        sequence_file: Path to .npy sequence file
    
    Returns:
        Prediction results
    """
    if not os.path.exists(sequence_file):
        raise FileNotFoundError(f"Sequence file not found: {sequence_file}")
    
    print(f"[INFO] Loading sequence from {sequence_file}")
    sequence = np.load(sequence_file)
    
    print(f"[INFO] Sequence shape: {sequence.shape}")
    
    # Make prediction
    predicted_class, confidence, class_probabilities = predict_gesture(
        model=model_artifacts['model'],
        label_encoder=model_artifacts['label_encoder'],
        train_mean=model_artifacts['train_mean'],
        train_std=model_artifacts['train_std'],
        sequence=sequence
    )
    
    return predicted_class, confidence, class_probabilities


def predict_from_sequence(model_artifacts, sequence):
    """
    Predict gesture from a numpy sequence array.
    
    Args:
        model_artifacts: Dictionary containing model and preprocessing artifacts
        sequence: Numpy array of shape (timesteps, features)
    
    Returns:
        Prediction results
    """
    print(f"[INFO] Predicting from sequence with shape: {sequence.shape}")
    
    # Make prediction
    predicted_class, confidence, class_probabilities = predict_gesture(
        model=model_artifacts['model'],
        label_encoder=model_artifacts['label_encoder'],
        train_mean=model_artifacts['train_mean'],
        train_std=model_artifacts['train_std'],
        sequence=sequence
    )
    
    return predicted_class, confidence, class_probabilities


def batch_predict(model_artifacts, data_dir, output_file=None):
    """
    Predict gestures for all .npy files in a directory.
    
    Args:
        model_artifacts: Dictionary containing model and preprocessing artifacts
        data_dir: Directory containing .npy sequence files
        output_file: Optional file to save results
    
    Returns:
        List of prediction results
    """
    import glob
    
    print(f"[INFO] Running batch predictions on {data_dir}")
    
    # Find all .npy files
    npy_files = glob.glob(os.path.join(data_dir, "*.npy"))
    
    if not npy_files:
        print(f"[WARNING] No .npy files found in {data_dir}")
        return []
    
    results = []
    correct_predictions = 0
    
    for file_path in sorted(npy_files):
        filename = os.path.basename(file_path)
        true_class = filename.split('_')[0]  # Extract class from filename
        
        try:
            predicted_class, confidence, class_probabilities = predict_from_file(model_artifacts, file_path)
            
            is_correct = predicted_class == true_class
            if is_correct:
                correct_predictions += 1
            
            result = {
                'filename': filename,
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct,
                'all_probabilities': class_probabilities
            }
            
            results.append(result)
            
            status = "✓" if is_correct else "✗"
            print(f"{status} {filename}: {true_class} -> {predicted_class} ({confidence:.3f})")
            
        except Exception as e:
            print(f"[ERROR] Failed to predict {filename}: {e}")
    
    # Calculate accuracy
    if results:
        accuracy = correct_predictions / len(results)
        print(f"\n[RESULT] Batch prediction accuracy: {accuracy*100:.2f}% ({correct_predictions}/{len(results)})")
    
    # Save results if requested
    if output_file:
        import json
        # Convert numpy values to regular Python types for JSON serialization
        json_results = []
        for result in results:
            json_result = {
                'filename': result['filename'],
                'true_class': result['true_class'],
                'predicted_class': result['predicted_class'],
                'confidence': float(result['confidence']),
                'correct': bool(result['correct']),
                'all_probabilities': {k: float(v) for k, v in result['all_probabilities'].items()}
            }
            json_results.append(json_result)
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"[INFO] Results saved to {output_file}")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict hand gestures using trained GRU model")
    parser.add_argument("--model_dir", type=str, default="model/gru",
                       help="Directory containing trained model (default: model/gru)")
    
    # Subparsers for different prediction modes
    subparsers = parser.add_subparsers(dest="mode", help="Prediction mode")
    
    # Single file prediction
    file_parser = subparsers.add_parser("file", help="Predict from single .npy file")
    file_parser.add_argument("sequence_file", type=str, help="Path to .npy sequence file")
    
    # Batch prediction
    batch_parser = subparsers.add_parser("batch", help="Predict from all .npy files in directory")
    batch_parser.add_argument("data_dir", type=str, help="Directory containing .npy files")
    batch_parser.add_argument("--output", type=str, help="Optional JSON file to save results")
    
    # Interactive mode
    interactive_parser = subparsers.add_parser("interactive", help="Interactive prediction mode")
    
    args = parser.parse_args()
    
    if args.mode is None:
        parser.print_help()
        return
    
    try:
        # Load trained model
        model_artifacts = load_trained_model(args.model_dir)
        
        if args.mode == "file":
            # Single file prediction
            predicted_class, confidence, class_probabilities = predict_from_file(
                model_artifacts, args.sequence_file
            )
            
            print(f"\n[RESULT] Prediction Results:")
            print(f"  File: {os.path.basename(args.sequence_file)}")
            print(f"  Predicted Class: {predicted_class}")
            print(f"  Confidence: {confidence:.3f}")
            print(f"  All Probabilities:")
            for class_name, prob in class_probabilities.items():
                print(f"    {class_name}: {prob:.3f}")
        
        elif args.mode == "batch":
            # Batch prediction
            results = batch_predict(model_artifacts, args.data_dir, args.output)
        
        elif args.mode == "interactive":
            # Interactive mode
            print(f"\n[INFO] Interactive prediction mode")
            print(f"Enter path to .npy file (or 'quit' to exit):")
            
            while True:
                try:
                    file_path = input("> ").strip()
                    
                    if file_path.lower() in ['quit', 'exit', 'q']:
                        break
                    
                    if not file_path:
                        continue
                    
                    predicted_class, confidence, class_probabilities = predict_from_file(
                        model_artifacts, file_path
                    )
                    
                    print(f"Predicted: {predicted_class} (confidence: {confidence:.3f})")
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            print("Goodbye!")
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()