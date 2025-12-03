"""
Real-time inference for multimodal gesture recognition.
Captures LMC and RGB data simultaneously and predicts gestures.
"""

import os
import sys
import argparse
import time
import json
import numpy as np
from collections import deque
from typing import Optional, List, Dict

import torch
import cv2

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.gru.multimodal_gru import create_multimodal_gru_model
from lmc.lmc_collector import LMCDataCollector
from rgb.rgb_collector import RGBDataCollector
from utils.geometry_features import extract_all_geometric_features
from utils.landmark_utils import landmarks_to_features


class RealtimeInference:
    """Real-time gesture recognition from LMC and RGB streams."""
    
    def __init__(
        self,
        model_path: str,
        label_map_path: str,
        sequence_length: int = 30,
        lmc_input_dim: int = 115,
        rgb_input_dim: int = 189,
        device: str = 'cuda',
        confidence_threshold: float = 0.5
    ):
        """
        Initialize real-time inference.
        
        Args:
            model_path: Path to trained model checkpoint
            label_map_path: Path to label mapping JSON file
            sequence_length: Sequence length for model
            lmc_input_dim: LMC input dimension
            rgb_input_dim: RGB input dimension
            device: Device to use
            confidence_threshold: Minimum confidence for prediction
        """
        self.sequence_length = sequence_length
        self.lmc_input_dim = lmc_input_dim
        self.rgb_input_dim = rgb_input_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        
        # Load label mapping
        with open(label_map_path, 'r') as f:
            label_data = json.load(f)
        self.idx_to_label = label_data['idx_to_label']
        self.num_classes = len(self.idx_to_label)
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Feature buffers
        self.lmc_buffer = deque(maxlen=sequence_length)
        self.rgb_buffer = deque(maxlen=sequence_length)
        
        # Prediction history
        self.prediction_history = deque(maxlen=10)
        
        # Data collectors
        self.lmc_collector = None
        self.rgb_collector = None
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract model configuration from checkpoint or use defaults
        model = create_multimodal_gru_model(
            num_classes=self.num_classes,
            lmc_input_dim=self.lmc_input_dim,
            rgb_input_dim=self.rgb_input_dim
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        print(f"Loaded model from {model_path}")
        print(f"Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A')}")
        
        return model
    
    def initialize_collectors(self, camera_id: int = 0) -> bool:
        """
        Initialize LMC and RGB data collectors.
        
        Args:
            camera_id: Camera device ID
        
        Returns:
            True if both collectors initialized successfully
        """
        # Initialize LMC collector
        self.lmc_collector = LMCDataCollector(include_geometric_features=True)
        lmc_connected = self.lmc_collector.connect()
        
        if not lmc_connected:
            print("Failed to connect to Leap Motion Controller")
            return False
        
        # Initialize RGB collector
        self.rgb_collector = RGBDataCollector(camera_id=camera_id, extract_features=True)
        rgb_connected = self.rgb_collector.connect()
        
        if not rgb_connected:
            print("Failed to connect to camera")
            self.lmc_collector.disconnect()
            return False
        
        print("Successfully initialized both collectors")
        return True
    
    def collect_features(self) -> Optional[tuple]:
        """
        Collect features from both modalities.
        
        Returns:
            Tuple of (lmc_features, rgb_features) or None if collection fails
        """
        # Collect LMC frame
        lmc_frame = self.lmc_collector.collect_frame()
        if lmc_frame is None or lmc_frame.get('features') is None:
            return None
        
        # Collect RGB frame
        rgb_frame = self.rgb_collector.collect_frame()
        if rgb_frame is None or rgb_frame.get('features') is None:
            return None
        
        lmc_features = np.array(lmc_frame['features'])
        rgb_features = np.array(rgb_frame['features'])
        
        return lmc_features, rgb_features
    
    def predict(self) -> Optional[Dict]:
        """
        Make prediction from current feature buffers.
        
        Returns:
            Dictionary with prediction results or None if buffer not full
        """
        if len(self.lmc_buffer) < self.sequence_length:
            return None
        
        # Prepare input tensors
        lmc_seq = np.array(list(self.lmc_buffer))
        rgb_seq = np.array(list(self.rgb_buffer))
        
        lmc_tensor = torch.FloatTensor(lmc_seq).unsqueeze(0).to(self.device)
        rgb_tensor = torch.FloatTensor(rgb_seq).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            probabilities = self.model.predict(lmc_tensor, rgb_tensor)
        
        # Get top prediction
        confidence, predicted_idx = probabilities.max(1)
        confidence = confidence.item()
        predicted_idx = predicted_idx.item()
        
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return {
                'gesture': 'Unknown',
                'confidence': confidence,
                'all_probabilities': probabilities[0].cpu().numpy()
            }
        
        gesture_label = self.idx_to_label.get(str(predicted_idx), 'Unknown')
        
        return {
            'gesture': gesture_label,
            'confidence': confidence,
            'predicted_idx': predicted_idx,
            'all_probabilities': probabilities[0].cpu().numpy()
        }
    
    def run(self, display: bool = True, save_video: Optional[str] = None):
        """
        Run real-time inference.
        
        Args:
            display: Whether to display video feed
            save_video: Path to save output video (optional)
        """
        if not self.initialize_collectors():
            print("Failed to initialize collectors")
            return
        
        print("\nStarting real-time inference...")
        print("Press 'q' to quit\n")
        
        # Video writer
        video_writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(save_video, fourcc, 30.0, (640, 480))
        
        current_gesture = "Waiting..."
        current_confidence = 0.0
        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0
        
        try:
            while True:
                # Collect features
                features = self.collect_features()
                
                if features is not None:
                    lmc_features, rgb_features = features
                    
                    # Add to buffers
                    self.lmc_buffer.append(lmc_features)
                    self.rgb_buffer.append(rgb_features)
                    
                    # Make prediction
                    if len(self.lmc_buffer) == self.sequence_length:
                        prediction = self.predict()
                        
                        if prediction is not None:
                            current_gesture = prediction['gesture']
                            current_confidence = prediction['confidence']
                            
                            # Smooth predictions
                            self.prediction_history.append(current_gesture)
                            
                            # Print prediction
                            print(f"Gesture: {current_gesture} (Confidence: {current_confidence:.2f})")
                
                # Display
                if display:
                    ret, frame = self.rgb_collector.cap.read()
                    if ret:
                        # Draw prediction
                        cv2.putText(frame, f"Gesture: {current_gesture}", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                        cv2.putText(frame, f"Confidence: {current_confidence:.2f}", (10, 70),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Draw FPS
                        frame_count += 1
                        if frame_count % 30 == 0:
                            fps = 30.0 / (time.time() - fps_start_time)
                            fps_start_time = time.time()
                        
                        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 450),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Buffer status
                        buffer_status = f"Buffer: {len(self.lmc_buffer)}/{self.sequence_length}"
                        cv2.putText(frame, buffer_status, (500, 450),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Show frame
                        cv2.imshow('Real-time Gesture Recognition', frame)
                        
                        # Save video
                        if video_writer:
                            video_writer.write(frame)
                        
                        # Check for quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                # Small delay
                time.sleep(0.01)
        
        finally:
            # Cleanup
            self.lmc_collector.disconnect()
            self.rgb_collector.disconnect()
            if video_writer:
                video_writer.release()
            cv2.destroyAllWindows()
            
            print("\nInference stopped.")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Real-time multimodal gesture recognition')
    
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--labels', type=str, required=True, help='Path to label mapping JSON')
    parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length')
    parser.add_argument('--lmc-input-dim', type=int, default=115, help='LMC input dimension')
    parser.add_argument('--rgb-input-dim', type=int, default=189, help='RGB input dimension')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--confidence-threshold', type=float, default=0.5, help='Confidence threshold')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--save-video', type=str, default=None, help='Save output video')
    
    args = parser.parse_args()
    
    # Create inference engine
    inference = RealtimeInference(
        model_path=args.model,
        label_map_path=args.labels,
        sequence_length=args.sequence_length,
        lmc_input_dim=args.lmc_input_dim,
        rgb_input_dim=args.rgb_input_dim,
        device=args.device,
        confidence_threshold=args.confidence_threshold
    )
    
    # Run inference
    inference.run(display=not args.no_display, save_video=args.save_video)


if __name__ == '__main__':
    main()
