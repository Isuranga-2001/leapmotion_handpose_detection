"""
Unified data collection script for simultaneous LMC and RGB capture.
Records synchronized multimodal data for gesture recognition.
"""

import os
import sys
import argparse
import time
import json
from typing import Dict, List

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from lmc.lmc_collector import LMCDataCollector
from rgb.rgb_collector import RGBDataCollector
from utils.sync import synchronize_streams, save_synchronized_data


def collect_synchronized_gesture(
    gesture_name: str,
    output_dir: str,
    duration: float = 5.0,
    fps: float = 30.0,
    camera_id: int = 0,
    display: bool = True
) -> Dict:
    """
    Collect synchronized LMC and RGB data for a gesture.
    
    Args:
        gesture_name: Name of the gesture
        output_dir: Output directory
        duration: Recording duration in seconds
        fps: Target frame rate
        camera_id: Camera device ID
        display: Whether to display video feed
    
    Returns:
        Dictionary with collection results
    """
    print(f"\n{'='*50}")
    print(f"Collecting gesture: {gesture_name}")
    print(f"Duration: {duration}s | FPS: {fps}")
    print(f"{'='*50}\n")
    
    # Create output directory
    gesture_dir = os.path.join(output_dir, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    
    # Initialize collectors
    print("Initializing collectors...")
    lmc_collector = LMCDataCollector(include_geometric_features=True)
    rgb_collector = RGBDataCollector(camera_id=camera_id, extract_features=True, save_images=False)
    
    # Connect
    if not lmc_collector.connect():
        print("Failed to connect to Leap Motion Controller")
        return {'success': False, 'error': 'LMC connection failed'}
    
    if not rgb_collector.connect():
        print("Failed to connect to camera")
        lmc_collector.disconnect()
        return {'success': False, 'error': 'Camera connection failed'}
    
    print("✓ Collectors initialized successfully\n")
    
    # Countdown
    print("Get ready! Recording will start in:")
    for i in range(3, 0, -1):
        print(f"{i}...")
        time.sleep(1)
    print("GO!\n")
    
    # Start recording
    start_time = time.time()
    
    # Record LMC data
    print("Recording LMC data...")
    lmc_frames = lmc_collector.record_sequence(duration=duration, fps=fps)
    
    # Record RGB data simultaneously (approximate)
    print("Recording RGB data...")
    rgb_frames = rgb_collector.record_sequence(duration=duration, fps=fps, display=display)
    
    recording_time = time.time() - start_time
    
    # Cleanup
    lmc_collector.disconnect()
    rgb_collector.disconnect()
    
    print(f"\n✓ Recording complete! ({recording_time:.2f}s)")
    print(f"LMC frames: {len(lmc_frames)}")
    print(f"RGB frames: {len(rgb_frames)}")
    
    # Check if we got data
    if len(lmc_frames) == 0 or len(rgb_frames) == 0:
        return {'success': False, 'error': 'No frames captured'}
    
    # Generate unique filename
    timestamp = int(time.time())
    base_filename = f"{gesture_name}_{timestamp}"
    
    # Save raw data
    lmc_file = os.path.join(gesture_dir, f"{base_filename}_lmc.json")
    rgb_file = os.path.join(gesture_dir, f"{base_filename}_rgb.json")
    
    lmc_collector.save_recording(lmc_file, lmc_frames)
    rgb_collector.save_recording(rgb_file, rgb_frames)
    
    # Synchronize streams
    print("\nSynchronizing streams...")
    try:
        synchronized_frames = synchronize_streams(
            lmc_frames,
            rgb_frames,
            max_time_diff=0.05,
            target_fps=fps
        )
        
        print(f"✓ Synchronized: {len(synchronized_frames)} frames")
        
        # Save synchronized data
        sync_file = os.path.join(gesture_dir, f"{base_filename}_synchronized.json")
        save_synchronized_data(synchronized_frames, sync_file)
        
        return {
            'success': True,
            'gesture': gesture_name,
            'lmc_frames': len(lmc_frames),
            'rgb_frames': len(rgb_frames),
            'synchronized_frames': len(synchronized_frames),
            'lmc_file': lmc_file,
            'rgb_file': rgb_file,
            'synchronized_file': sync_file,
            'duration': recording_time
        }
    
    except Exception as e:
        print(f"Error during synchronization: {e}")
        return {
            'success': False,
            'error': f'Synchronization failed: {e}',
            'lmc_file': lmc_file,
            'rgb_file': rgb_file
        }


def collect_dataset(
    gestures: List[str],
    output_dir: str,
    samples_per_gesture: int = 10,
    duration: float = 5.0,
    fps: float = 30.0,
    camera_id: int = 0
):
    """
    Collect a complete dataset with multiple gestures.
    
    Args:
        gestures: List of gesture names
        output_dir: Output directory
        samples_per_gesture: Number of samples to collect per gesture
        duration: Recording duration per sample
        fps: Target frame rate
        camera_id: Camera device ID
    """
    print("\n" + "="*60)
    print("MULTIMODAL GESTURE DATASET COLLECTION")
    print("="*60)
    print(f"Gestures: {', '.join(gestures)}")
    print(f"Samples per gesture: {samples_per_gesture}")
    print(f"Total samples: {len(gestures) * samples_per_gesture}")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Collection log
    collection_log = []
    
    # Collect data for each gesture
    for gesture_idx, gesture in enumerate(gestures):
        print(f"\n{'='*60}")
        print(f"Gesture {gesture_idx + 1}/{len(gestures)}: {gesture}")
        print(f"{'='*60}")
        
        for sample_idx in range(samples_per_gesture):
            print(f"\n--- Sample {sample_idx + 1}/{samples_per_gesture} ---")
            
            # Collect sample
            result = collect_synchronized_gesture(
                gesture_name=gesture,
                output_dir=output_dir,
                duration=duration,
                fps=fps,
                camera_id=camera_id,
                display=True
            )
            
            # Log result
            collection_log.append(result)
            
            if result['success']:
                print(f"✓ Sample collected successfully")
            else:
                print(f"✗ Sample collection failed: {result.get('error', 'Unknown error')}")
            
            # Rest between samples
            if sample_idx < samples_per_gesture - 1:
                print("\nResting for 3 seconds...")
                time.sleep(3)
    
    # Save collection log
    log_file = os.path.join(output_dir, 'collection_log.json')
    with open(log_file, 'w') as f:
        json.dump(collection_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total samples collected: {len([r for r in collection_log if r['success']])}/{len(collection_log)}")
    print(f"Log saved to: {log_file}")
    print(f"{'='*60}\n")


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(description='Multimodal gesture data collection')
    
    parser.add_argument('--gesture', type=str, help='Single gesture name')
    parser.add_argument('--gestures', type=str, nargs='+', help='Multiple gesture names')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--samples', type=int, default=1, help='Samples per gesture')
    parser.add_argument('--duration', type=float, default=5.0, help='Recording duration (seconds)')
    parser.add_argument('--fps', type=float, default=30.0, help='Target FPS')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Determine gestures to collect
    if args.gesture:
        gestures = [args.gesture]
    elif args.gestures:
        gestures = args.gestures
    else:
        print("Error: Must specify either --gesture or --gestures")
        return
    
    # Collect dataset
    collect_dataset(
        gestures=gestures,
        output_dir=args.output_dir,
        samples_per_gesture=args.samples,
        duration=args.duration,
        fps=args.fps,
        camera_id=args.camera
    )


if __name__ == '__main__':
    main()
