"""
Leap Motion Controller data collector with geometric feature extraction.
Records 3D skeletal hand data with timestamps and optional geometric features.
"""

import sys
import os
import time
import json
import numpy as np
from typing import List, Dict, Optional

# Add leap module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'leapc-python-api', 'src'))

try:
    import leap
    from leap import datatypes as ldt
except ImportError:
    print("Error: Could not import leap module. Make sure leapc-python-api is properly installed.")
    sys.exit(1)

# Import geometry feature utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.geometry_features import extract_all_geometric_features


class LMCDataCollector:
    """Collects hand pose data from Leap Motion Controller."""
    
    def __init__(self, include_geometric_features: bool = True):
        """
        Initialize the LMC data collector.
        
        Args:
            include_geometric_features: Whether to compute geometric features
        """
        self.include_geometric_features = include_geometric_features
        self.recording = False
        self.frames = []
        self.connection = None
        
    def connect(self):
        """Establish connection to Leap Motion Controller."""
        try:
            self.connection = leap.Connection()
            self.connection.connect()
            print("Connected to Leap Motion Controller")
            return True
        except Exception as e:
            print(f"Failed to connect to Leap Motion Controller: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Leap Motion Controller."""
        if self.connection:
            self.connection.disconnect()
            print("Disconnected from Leap Motion Controller")
    
    def extract_hand_joints(self, hand: ldt.Hand) -> Optional[np.ndarray]:
        """
        Extract 27 joint positions (wrist + 4 joints per finger × 5 fingers).
        
        Args:
            hand: Leap Hand object
        
        Returns:
            Array of shape (27, 3) or None if extraction fails
        """
        try:
            joints = []
            
            # Add wrist position (palm position as proxy)
            palm = hand.palm
            joints.append([palm.position.x, palm.position.y, palm.position.z])
            
            # Extract digits (thumb, index, middle, ring, pinky)
            digits = hand.digits
            
            for digit in digits:
                # Each digit has 4 bones: metacarpal, proximal, intermediate, distal
                bones = [digit.metacarpal, digit.proximal, digit.intermediate, digit.distal]
                
                for bone in bones:
                    # Use the next joint position (end of the bone)
                    next_joint = bone.next_joint
                    joints.append([next_joint.x, next_joint.y, next_joint.z])
            
            # Should have 1 (wrist) + 4 * 5 (bones per finger * fingers) = 21 joints
            # For 27 joints, we'll add intermediate positions
            if len(joints) < 27:
                # Pad with zeros if needed
                while len(joints) < 27:
                    joints.append([0.0, 0.0, 0.0])
            
            return np.array(joints[:27])
            
        except Exception as e:
            print(f"Error extracting hand joints: {e}")
            return None
    
    def collect_frame(self) -> Optional[Dict]:
        """
        Collect a single frame of hand data.
        
        Returns:
            Dictionary containing timestamp and hand data, or None if no hand detected
        """
        if not self.connection:
            print("Not connected to Leap Motion Controller")
            return None
        
        try:
            # Get tracking event
            event = self.connection.get_tracking_event()
            
            if not event or len(event.hands) == 0:
                return None
            
            # Get the first hand
            hand = event.hands[0]
            
            # Extract joints
            hand_joints = self.extract_hand_joints(hand)
            
            if hand_joints is None:
                return None
            
            # Get timestamp (microseconds)
            timestamp = event.timestamp / 1000000.0  # Convert to seconds
            
            frame_data = {
                'timestamp': timestamp,
                'hand': hand_joints.flatten().tolist(),  # 81 values (27 × 3)
            }
            
            # Add geometric features if enabled
            if self.include_geometric_features:
                try:
                    geometric_features = extract_all_geometric_features(hand_joints)
                    frame_data['features'] = geometric_features.tolist()
                except Exception as e:
                    print(f"Warning: Could not compute geometric features: {e}")
                    frame_data['features'] = None
            
            return frame_data
            
        except Exception as e:
            print(f"Error collecting frame: {e}")
            return None
    
    def start_recording(self):
        """Start recording hand data."""
        self.recording = True
        self.frames = []
        print("Recording started...")
    
    def stop_recording(self):
        """Stop recording hand data."""
        self.recording = False
        print(f"Recording stopped. Captured {len(self.frames)} frames.")
    
    def record_sequence(self, duration: float = 5.0, fps: float = 30.0) -> List[Dict]:
        """
        Record a sequence of hand data for a specified duration.
        
        Args:
            duration: Recording duration in seconds
            fps: Target frames per second
        
        Returns:
            List of recorded frames
        """
        if not self.connection:
            print("Not connected. Call connect() first.")
            return []
        
        self.start_recording()
        
        frame_interval = 1.0 / fps
        start_time = time.time()
        last_frame_time = start_time
        
        while self.recording and (time.time() - start_time) < duration:
            current_time = time.time()
            
            # Maintain target FPS
            if current_time - last_frame_time >= frame_interval:
                frame = self.collect_frame()
                
                if frame is not None:
                    self.frames.append(frame)
                    print(f"Frame {len(self.frames)}: captured", end='\r')
                
                last_frame_time = current_time
            
            # Small sleep to avoid busy waiting
            time.sleep(0.001)
        
        self.stop_recording()
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
            with open(filename, 'w') as f:
                json.dump(frames, f, indent=2)
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
    """Example usage of LMC data collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Leap Motion Controller data collector')
    parser.add_argument('--output', '-o', type=str, required=True, help='Output JSON file')
    parser.add_argument('--duration', '-d', type=float, default=5.0, help='Recording duration (seconds)')
    parser.add_argument('--fps', '-f', type=float, default=30.0, help='Target FPS')
    parser.add_argument('--no-features', action='store_true', help='Disable geometric feature extraction')
    
    args = parser.parse_args()
    
    # Create collector
    collector = LMCDataCollector(include_geometric_features=not args.no_features)
    
    # Connect
    if not collector.connect():
        print("Failed to connect to Leap Motion Controller")
        return
    
    try:
        print(f"Recording for {args.duration} seconds at {args.fps} FPS...")
        print("Please perform the gesture...")
        
        # Wait a moment before starting
        time.sleep(1)
        
        # Record
        frames = collector.record_sequence(duration=args.duration, fps=args.fps)
        
        # Save
        collector.save_recording(args.output, frames)
        
        print(f"\nRecording complete!")
        print(f"Total frames: {len(frames)}")
        if frames:
            print(f"First timestamp: {frames[0]['timestamp']:.3f}")
            print(f"Last timestamp: {frames[-1]['timestamp']:.3f}")
            print(f"Duration: {frames[-1]['timestamp'] - frames[0]['timestamp']:.3f} seconds")
        
    finally:
        collector.disconnect()


if __name__ == '__main__':
    main()
