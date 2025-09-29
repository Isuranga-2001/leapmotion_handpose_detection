#!/usr/bin/env python3
"""
Example script demonstrating real-time gesture recognition using the trained GRU model.
This script shows how to integrate the GRU model with LeapMotion for live gesture classification.
"""

import numpy as np
import time
import sys
import os
import cv2
import threading

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'functions'))

# Add current directory to Python path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    import leap
    from functions.extract_features import extract_features
    from model_train import load_model_artifacts, predict_gesture
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    
    # Try alternative import for model_train
    try:
        import importlib.util
        model_train_path = os.path.join(current_dir, 'model_train.py')
        if os.path.exists(model_train_path):
            spec = importlib.util.spec_from_file_location("model_train", model_train_path)
            model_train = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(model_train)
            load_model_artifacts = model_train.load_model_artifacts
            predict_gesture = model_train.predict_gesture
        else:
            raise ImportError("Could not find model_train.py")
    except Exception as inner_e:
        print(f"[ERROR] Failed to import required modules: {inner_e}")
        print("Make sure all dependencies are installed and paths are correct.")
        sys.exit(1)


class PredictionDisplay:
    """
    Simple OpenCV window to display gesture predictions.
    """
    
    def __init__(self):
        self.window_name = "Gesture Prediction"
        self.current_prediction = "No prediction"
        self.confidence = 0.0
        self.is_running = False
        
        # Create a blank image for display
        self.img_height = 300
        self.img_width = 600
        self.display_thread = None
        
    def start(self):
        """Start the display in a separate thread."""
        self.is_running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.daemon = True
        self.display_thread.start()
        
    def stop(self):
        """Stop the display."""
        self.is_running = False
        cv2.destroyAllWindows()
        
    def update_prediction(self, prediction, confidence):
        """Update the prediction display."""
        self.current_prediction = prediction if prediction else "No prediction"
        self.confidence = confidence if confidence else 0.0
        
    def _display_loop(self):
        """Main display loop running in separate thread."""
        while self.is_running:
            # Create blank image
            img = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            
            # Add title
            cv2.putText(img, "Gesture Recognition", (150, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
            
            # Add current prediction
            cv2.putText(img, "Prediction:", (50, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Color code the prediction based on confidence
            if self.confidence > 0.7:
                color = (0, 255, 0)  # Green for high confidence
            elif self.confidence > 0.5:
                color = (0, 255, 255)  # Yellow for medium confidence
            else:
                color = (100, 100, 100)  # Gray for low/no confidence
                
            cv2.putText(img, self.current_prediction, (50, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Add confidence
            # if self.confidence > 0:
            #     confidence_text = f"Confidence: {self.confidence:.3f}"
            #     cv2.putText(img, confidence_text, (50, 220), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            # Add instructions
            cv2.putText(img, "Press 'q' to quit", (50, 270), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            
            # Show the image
            cv2.imshow(self.window_name, img)
            
            # Check for quit key
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                self.is_running = False
                break
                
        cv2.destroyAllWindows()


class GRUGestureRecognizer:
    """
    Real-time gesture recognition using trained GRU model.
    """
    
    def __init__(self, model_dir=None, sequence_length=50, prediction_threshold=0.7):
        """
        Initialize the GRU gesture recognizer.
        
        Args:
            model_dir: Directory containing trained model artifacts
            sequence_length: Length of sequences to maintain for prediction
            prediction_threshold: Minimum confidence threshold for predictions
        """
        # Auto-detect model directory if not provided
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.model_dir = model_dir
        self.sequence_length = sequence_length
        self.prediction_threshold = prediction_threshold
        
        # Load trained model
        print(f"[INFO] Loading GRU model from {model_dir}")
        try:
            self.model_artifacts = load_model_artifacts(
                os.path.join(model_dir, "model_artifacts.pkl")
            )
            print(f"[INFO] Model loaded successfully!")
            print(f"[INFO] Available classes: {list(self.model_artifacts['label_encoder'].classes_)}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
        
        # Initialize sequence buffer
        self.sequence_buffer = []
        self.last_prediction = None
        self.last_prediction_time = 0
        self.prediction_interval = 0.5  # Predict every 0.5 seconds
        
    def add_frame(self, hand):
        """
        Add a new frame to the sequence buffer.
        
        Args:
            hand: LeapMotion hand object
        """
        # Extract features from hand
        features = extract_features(hand)
        
        # Add timestamp and hand type (consistent with training data)
        features['timestamp'] = time.time()
        features['hand_type'] = 1 if hand.type == leap.HandType.Right else 0
        
        # Convert to feature vector (same order as training)
        feature_vector = list(features.values())
        
        # Add to buffer
        self.sequence_buffer.append(feature_vector)
        
        # Maintain buffer size
        if len(self.sequence_buffer) > self.sequence_length:
            self.sequence_buffer.pop(0)
    
    def predict_gesture(self):
        """
        Predict gesture from current sequence buffer.
        
        Returns:
            tuple: (predicted_class, confidence, all_probabilities) or (None, None, None)
        """
        # Check if we have enough frames
        if len(self.sequence_buffer) < self.sequence_length:
            return None, None, None
        
        # Check prediction interval
        current_time = time.time()
        if current_time - self.last_prediction_time < self.prediction_interval:
            return self.last_prediction
        
        try:
            # Convert buffer to numpy array
            sequence = np.array(self.sequence_buffer[-self.sequence_length:], dtype=np.float32)
            
            # Make prediction
            predicted_class, confidence, class_probabilities = predict_gesture(
                model=self.model_artifacts['model'],
                label_encoder=self.model_artifacts['label_encoder'],
                train_mean=self.model_artifacts['train_mean'],
                train_std=self.model_artifacts['train_std'],
                sequence=sequence
            )
            
            # Apply confidence threshold
            if confidence < self.prediction_threshold:
                result = None, confidence, class_probabilities
            else:
                result = predicted_class, confidence, class_probabilities
            
            self.last_prediction = result
            self.last_prediction_time = current_time
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {e}")
            return None, None, None
    
    def reset_buffer(self):
        """Reset the sequence buffer."""
        self.sequence_buffer = []
        self.last_prediction = None


class GRUListener(leap.Listener):
    """
    LeapMotion listener for real-time gesture recognition with GRU model.
    """
    
    def __init__(self, model_dir=None, prediction_display=None):
        super().__init__()
        # Auto-detect model directory if not provided
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))
        self.recognizer = GRUGestureRecognizer(model_dir)
        self.frame_count = 0
        self.prediction_display = prediction_display
        
    def on_connection_event(self, event):
        print("[INFO] Connected to Leap Motion device!")
        
    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"[INFO] Found device: {info.serial}")
        
    def on_tracking_event(self, event):
        self.frame_count += 1
        
        # Process only every few frames to reduce computation
        if self.frame_count % 3 != 0:
            return
        
        if not event.hands:
            # Update display to show no hand detected
            if self.prediction_display:
                self.prediction_display.update_prediction("No hand detected", 0.0)
            return
        
        # Process the first detected hand
        hand = event.hands[0]
        
        # Add frame to recognizer
        self.recognizer.add_frame(hand)
        
        # Try to predict gesture
        predicted_class, confidence, class_probabilities = self.recognizer.predict_gesture()
        
        if predicted_class is not None:
            print(f"[PREDICTION] {predicted_class} (confidence: {confidence:.3f})")
            
            # Show all class probabilities
            if class_probabilities:
                prob_str = ", ".join([f"{cls}: {prob:.3f}" for cls, prob in class_probabilities.items()])
                print(f"[PROBABILITIES] {prob_str}")
            
            # Update display
            if self.prediction_display:
                self.prediction_display.update_prediction(predicted_class, confidence)
                
        elif confidence is not None:
            best_guess = max(class_probabilities.items(), key=lambda x: x[1])[0] if class_probabilities else "Unknown"
            print(f"[LOW CONFIDENCE] Best guess: {best_guess} ({confidence:.3f})")
            
            # Update display with low confidence prediction
            if self.prediction_display:
                self.prediction_display.update_prediction(f"{best_guess}", confidence)
        else:
            # Update display to show processing
            if self.prediction_display:
                self.prediction_display.update_prediction("Processing...", 0.0)


def main():
    """
    Main function to run real-time gesture recognition.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time gesture recognition with GRU model")
    
    # Auto-detect model directory based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = script_dir  # Model should be in the same directory as the script
    
    parser.add_argument("--model_dir", type=str, default=default_model_dir,
                       help=f"Directory containing trained GRU model (default: {default_model_dir})")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Prediction confidence threshold (default: 0.7)")
    
    args = parser.parse_args()
    
    # Check if model exists
    model_artifacts_path = os.path.join(args.model_dir, "model_artifacts.pkl")
    if not os.path.exists(model_artifacts_path):
        print(f"[ERROR] Model artifacts not found at: {model_artifacts_path}")
        print("Please train the model first using model_train.py")
        return
    
    try:
        # Create prediction display
        prediction_display = PredictionDisplay()
        prediction_display.start()
        
        # Create listener with GRU recognizer
        listener = GRUListener(args.model_dir, prediction_display)
        listener.recognizer.prediction_threshold = args.threshold
        
        # Setup Leap Motion connection
        connection = leap.Connection()
        connection.add_listener(listener)
        
        with connection.open():
            print(f"\n[INFO] Starting real-time gesture recognition...")
            print(f"[INFO] Model directory: {args.model_dir}")
            print(f"[INFO] Confidence threshold: {args.threshold}")
            print(f"[INFO] Available classes: {list(listener.recognizer.model_artifacts['label_encoder'].classes_)}")
            print("\n[CONTROLS]")
            print("  - Place your hand over the Leap Motion sensor")
            print("  - Perform gestures to see real-time predictions")
            print("  - Press 'q' in the prediction window or Ctrl+C to exit")
            print("\nStarting recognition...\n")
            
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            
            try:
                # Keep running until display window is closed or Ctrl+C
                while prediction_display.is_running:
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n[INFO] Stopping gesture recognition...")
                
        # Clean up
        prediction_display.stop()
        print("[INFO] Gesture recognition stopped.")
        
    except Exception as e:
        print(f"[ERROR] Failed to start gesture recognition: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()