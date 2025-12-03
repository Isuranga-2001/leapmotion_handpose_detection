"""
RGB camera data collector with facial landmark and head pose extraction.
Uses OpenCV for video capture and MediaPipe for facial analysis.
"""

import cv2
import numpy as np
import time
import json
import sys
import os
from typing import List, Dict, Optional

# Import landmark utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.landmark_utils import FaceLandmarkExtractor, landmarks_to_features


class RGBDataCollector:
    """Collects RGB camera data with facial landmark and pose extraction."""
    
    def __init__(
        self,
        camera_id: int = 0,
        extract_features: bool = True,
        save_images: bool = False
    ):
        """
        Initialize the RGB data collector.
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            extract_features: Whether to extract facial features
            save_images: Whether to save captured images
        """
        self.camera_id = camera_id
        self.extract_features = extract_features
        self.save_images = save_images
        self.recording = False
        self.frames = []
        self.cap = None
        self.landmark_extractor = None
        self.image_dir = None
    
    def connect(self) -> bool:
        """Open connection to camera."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                print(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Initialize landmark extractor
            if self.extract_features:
                self.landmark_extractor = FaceLandmarkExtractor()
            
            print(f"Connected to camera {self.camera_id}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to camera: {e}")
            return False
    
    def disconnect(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()
            print("Camera released")
        
        if self.landmark_extractor is not None:
            self.landmark_extractor.close()
        
        cv2.destroyAllWindows()
    
    def collect_frame(self, frame_number: int = 0) -> Optional[Dict]:
        """
        Collect a single frame from the camera.
        
        Args:
            frame_number: Frame number for saving images
        
        Returns:
            Dictionary containing timestamp and facial data, or None if capture fails
        """
        if self.cap is None:
            print("Camera not connected")
            return None
        
        try:
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return None
            
            # Get timestamp
            timestamp = time.time()
            
            frame_data = {
                'timestamp': timestamp,
                'frame_number': frame_number
            }
            
            # Extract facial landmarks and pose
            if self.extract_features and self.landmark_extractor is not None:
                landmarks_data = self.landmark_extractor.extract_landmarks(frame)
                
                if landmarks_data is not None:
                    # Convert landmarks to feature vector
                    feature_vector = landmarks_to_features(landmarks_data)
                    
                    frame_data['facial_landmarks'] = landmarks_data['landmarks'].tolist()
                    frame_data['pose'] = landmarks_data['pose']
                    frame_data['features'] = feature_vector.tolist()
                else:
                    # No face detected
                    frame_data['facial_landmarks'] = None
                    frame_data['pose'] = None
                    frame_data['features'] = None
            
            # Save image if enabled
            if self.save_images and self.image_dir is not None:
                image_filename = os.path.join(self.image_dir, f'frame_{frame_number:06d}.jpg')
                cv2.imwrite(image_filename, frame)
                frame_data['image_path'] = image_filename
            
            return frame_data
            
        except Exception as e:
            print(f"Error collecting frame: {e}")
            return None
    
    def start_recording(self, image_dir: Optional[str] = None):
        """
        Start recording RGB data.
        
        Args:
            image_dir: Directory to save images (if save_images is True)
        """
        self.recording = True
        self.frames = []
        
        if self.save_images and image_dir is not None:
            self.image_dir = image_dir
            os.makedirs(image_dir, exist_ok=True)
        
        print("Recording started...")
    
    def stop_recording(self):
        """Stop recording RGB data."""
        self.recording = False
        print(f"Recording stopped. Captured {len(self.frames)} frames.")
    
    def record_sequence(
        self,
        duration: float = 5.0,
        fps: float = 30.0,
        display: bool = True,
        image_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Record a sequence of RGB data for a specified duration.
        
        Args:
            duration: Recording duration in seconds
            fps: Target frames per second
            display: Whether to display video feed
            image_dir: Directory to save images
        
        Returns:
            List of recorded frames
        """
        if self.cap is None:
            print("Not connected. Call connect() first.")
            return []
        
        self.start_recording(image_dir)
        
        frame_interval = 1.0 / fps
        start_time = time.time()
        last_frame_time = start_time
        frame_number = 0
        
        while self.recording and (time.time() - start_time) < duration:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time >= frame_interval:
                # Capture frame
                ret, frame = self.cap.read()
                
                if ret and frame is not None:
                    # Process frame
                    frame_data = self.collect_frame(frame_number)
                    
                    if frame_data is not None:
                        self.frames.append(frame_data)
                        frame_number += 1
                        print(f"Frame {len(self.frames)}: captured", end='\r')
                    
                    # Display frame
                    if display:
                        display_frame = frame.copy()
                        
                        # Draw face detection status
                        if frame_data and frame_data.get('facial_landmarks') is not None:
                            cv2.putText(display_frame, "Face Detected", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            # Optionally draw pose info
                            pose = frame_data.get('pose')
                            if pose:
                                text = f"Pitch: {pose['pitch']:.1f} Yaw: {pose['yaw']:.1f} Roll: {pose['roll']:.1f}"
                                cv2.putText(display_frame, text, (10, 60),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        else:
                            cv2.putText(display_frame, "No Face Detected", (10, 30),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Show remaining time
                        remaining = duration - (current_time - start_time)
                        cv2.putText(display_frame, f"Time: {remaining:.1f}s", (10, 450),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        cv2.imshow('RGB Collector', display_frame)
                        
                        # Check for exit key
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                last_frame_time = current_time
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
        
        self.stop_recording()
        
        if display:
            cv2.destroyAllWindows()
        
        return self.frames
    
    def save_recording(self, filename: str, frames: Optional[List[Dict]] = None):
        """
        Save recorded frames to a JSON file.
        
        Args:
            filename: Output filename
            frames: Frames to save (uses self.frames if None)
        """
        if frames is None:
            frames = self.frames
        
        if not frames:
            print("No frames to save")
            return
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_frames = []
            for frame in frames:
                frame_copy = frame.copy()
                if 'facial_landmarks' in frame_copy and frame_copy['facial_landmarks'] is not None:
                    if isinstance(frame_copy['facial_landmarks'], np.ndarray):
                        frame_copy['facial_landmarks'] = frame_copy['facial_landmarks'].tolist()
                serializable_frames.append(frame_copy)
            
            with open(filename, 'w') as f:
                json.dump(serializable_frames, f, indent=2)
            print(f"Saved {len(frames)} frames to {filename}")
        except Exception as e:
            print(f"Error saving recording: {e}")
    
    def load_recording(self, filename: str) -> List[Dict]:
        """
        Load recorded frames from a JSON file.
        
        Args:
            filename: Input filename
        
        Returns:
            List of frames
        """
        try:
            with open(filename, 'r') as f:
                frames = json.load(f)
            print(f"Loaded {len(frames)} frames from {filename}")
            return frames
        except Exception as e:
            print(f"Error loading recording: {e}")
            return []


def main():
    """Example usage of RGB data collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='RGB camera data collector')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSON file')
    parser.add_argument('--duration', '-d', type=float, default=5.0, help='Recording duration (seconds)')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Target FPS')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera ID')
    parser.add_argument('--save-images', action='store_true', help='Save captured images')
    parser.add_argument('--image-dir', type=str, default='./rgb_images', help='Directory to save images')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Create collector
    collector = RGBDataCollector(
        camera_id=args.camera,
        extract_features=True,
        save_images=args.save_images
    )
    
    # Connect
    if not collector.connect():
        print("Failed to connect to camera")
        return
    
    try:
        print(f"Recording for {args.duration} seconds at {args.fps} FPS...")
        print("Please position your face in the camera view...")
        print("Press 'q' to stop recording early")
        
        # Wait a moment before starting
        time.sleep(1)
        
        # Record
        image_dir = args.image_dir if args.save_images else None
        frames = collector.record_sequence(
            duration=args.duration,
            fps=args.fps,
            display=not args.no_display,
            image_dir=image_dir
        )
        
        # Save
        collector.save_recording(args.output, frames)
        
        print(f"\nRecording complete!")
        print(f"Total frames: {len(frames)}")
        if frames:
            frames_with_face = sum(1 for f in frames if f.get('facial_landmarks') is not None)
            print(f"Frames with face detected: {frames_with_face}/{len(frames)}")
            print(f"First timestamp: {frames[0]['timestamp']:.3f}")
            print(f"Last timestamp: {frames[-1]['timestamp']:.3f}")
            print(f"Duration: {frames[-1]['timestamp'] - frames[0]['timestamp']:.3f} seconds")
        
    finally:
        collector.disconnect()


if __name__ == '__main__':
    main()
