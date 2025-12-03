"""
Synchronization utilities for aligning LMC and RGB camera data streams.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.interpolate import interp1d
import json


def synchronize_streams(
    lmc_stream: List[Dict],
    rgb_stream: List[Dict],
    max_time_diff: float = 0.05,
    target_fps: Optional[float] = None
) -> List[Dict]:
    """
    Synchronize LMC and RGB data streams by matching timestamps.
    
    Args:
        lmc_stream: List of LMC frames with 'timestamp' and 'hand' keys
        rgb_stream: List of RGB frames with 'timestamp' and 'facial_landmarks'/'pose' keys
        max_time_diff: Maximum allowed time difference (in seconds) for matching
        target_fps: If specified, resample both streams to this FPS before matching
    
    Returns:
        List of synchronized frames containing both LMC and RGB data
    """
    if not lmc_stream or not rgb_stream:
        raise ValueError("Both streams must contain data")
    
    # Sort streams by timestamp
    lmc_stream = sorted(lmc_stream, key=lambda x: x['timestamp'])
    rgb_stream = sorted(rgb_stream, key=lambda x: x['timestamp'])
    
    # If target_fps is specified, resample streams
    if target_fps is not None:
        lmc_stream = resample_stream(lmc_stream, target_fps)
        rgb_stream = resample_stream(rgb_stream, target_fps)
    
    synchronized_frames = []
    rgb_idx = 0
    
    for lmc_frame in lmc_stream:
        lmc_time = lmc_frame['timestamp']
        
        # Find the closest RGB frame
        while rgb_idx < len(rgb_stream) - 1:
            current_diff = abs(rgb_stream[rgb_idx]['timestamp'] - lmc_time)
            next_diff = abs(rgb_stream[rgb_idx + 1]['timestamp'] - lmc_time)
            
            if next_diff < current_diff:
                rgb_idx += 1
            else:
                break
        
        rgb_frame = rgb_stream[rgb_idx]
        time_diff = abs(rgb_frame['timestamp'] - lmc_time)
        
        # Only include if timestamps are close enough
        if time_diff <= max_time_diff:
            synchronized_frame = {
                'timestamp': lmc_time,
                'lmc': lmc_frame.get('hand'),
                'lmc_features': lmc_frame.get('features'),
                'rgb': rgb_frame.get('facial_landmarks'),
                'rgb_features': rgb_frame.get('features'),
                'pose': rgb_frame.get('pose'),
                'time_diff': time_diff
            }
            synchronized_frames.append(synchronized_frame)
    
    return synchronized_frames


def interpolate_lmc_data(
    lmc_stream: List[Dict],
    target_timestamps: np.ndarray
) -> List[Dict]:
    """
    Interpolate LMC hand joint positions to match target timestamps.
    Useful for upsampling or downsampling LMC data to match RGB FPS.
    
    Args:
        lmc_stream: List of LMC frames with 'timestamp' and 'hand' keys
        target_timestamps: Array of target timestamps
    
    Returns:
        List of interpolated LMC frames
    """
    if not lmc_stream:
        raise ValueError("LMC stream is empty")
    
    # Extract timestamps and hand data
    timestamps = np.array([frame['timestamp'] for frame in lmc_stream])
    hand_data = np.array([frame['hand'] for frame in lmc_stream])  # Shape: (N, 81)
    
    if hand_data.ndim == 1:
        hand_data = hand_data.reshape(-1, 1)
    
    # Create interpolation functions for each joint coordinate
    interpolators = []
    for i in range(hand_data.shape[1]):
        interpolator = interp1d(
            timestamps,
            hand_data[:, i],
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        interpolators.append(interpolator)
    
    # Interpolate at target timestamps
    interpolated_frames = []
    for target_time in target_timestamps:
        if target_time < timestamps[0] or target_time > timestamps[-1]:
            continue  # Skip timestamps outside the range
        
        interpolated_hand = np.array([interp(target_time) for interp in interpolators])
        
        frame = {
            'timestamp': float(target_time),
            'hand': interpolated_hand.tolist()
        }
        interpolated_frames.append(frame)
    
    return interpolated_frames


def resample_stream(
    stream: List[Dict],
    target_fps: float
) -> List[Dict]:
    """
    Resample a data stream to a target frame rate.
    
    Args:
        stream: List of frames with 'timestamp' key
        target_fps: Target frames per second
    
    Returns:
        Resampled stream
    """
    if not stream:
        return []
    
    timestamps = np.array([frame['timestamp'] for frame in stream])
    start_time = timestamps[0]
    end_time = timestamps[-1]
    duration = end_time - start_time
    
    if duration <= 0:
        return stream
    
    # Generate target timestamps
    num_frames = int(duration * target_fps)
    target_timestamps = np.linspace(start_time, end_time, num_frames)
    
    # Check if stream contains hand data (LMC) or facial data (RGB)
    if 'hand' in stream[0]:
        return interpolate_lmc_data(stream, target_timestamps)
    else:
        # For RGB data, use nearest neighbor matching
        resampled_stream = []
        for target_time in target_timestamps:
            # Find nearest frame
            idx = np.argmin(np.abs(timestamps - target_time))
            frame = stream[idx].copy()
            frame['timestamp'] = float(target_time)
            resampled_stream.append(frame)
        return resampled_stream


def align_sequences_by_gesture(
    lmc_stream: List[Dict],
    rgb_stream: List[Dict],
    gesture_start_marker: str = "start",
    gesture_end_marker: str = "end"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Align sequences by gesture markers (start and end points).
    Useful when data collection includes manual markers.
    
    Args:
        lmc_stream: LMC data stream
        rgb_stream: RGB data stream
        gesture_start_marker: Key indicating gesture start
        gesture_end_marker: Key indicating gesture end
    
    Returns:
        Tuple of (aligned_lmc_stream, aligned_rgb_stream)
    """
    # Find start and end markers in both streams
    lmc_start = None
    lmc_end = None
    rgb_start = None
    rgb_end = None
    
    for i, frame in enumerate(lmc_stream):
        if frame.get('marker') == gesture_start_marker and lmc_start is None:
            lmc_start = i
        if frame.get('marker') == gesture_end_marker and lmc_end is None:
            lmc_end = i
    
    for i, frame in enumerate(rgb_stream):
        if frame.get('marker') == gesture_start_marker and rgb_start is None:
            rgb_start = i
        if frame.get('marker') == gesture_end_marker and rgb_end is None:
            rgb_end = i
    
    # Extract sequences between markers
    if lmc_start is not None and lmc_end is not None:
        aligned_lmc = lmc_stream[lmc_start:lmc_end + 1]
    else:
        aligned_lmc = lmc_stream
    
    if rgb_start is not None and rgb_end is not None:
        aligned_rgb = rgb_stream[rgb_start:rgb_end + 1]
    else:
        aligned_rgb = rgb_stream
    
    return aligned_lmc, aligned_rgb


def load_and_synchronize(
    lmc_file: str,
    rgb_file: str,
    target_fps: Optional[float] = 30.0,
    max_time_diff: float = 0.05
) -> List[Dict]:
    """
    Load LMC and RGB data from files and synchronize them.
    
    Args:
        lmc_file: Path to LMC JSON file
        rgb_file: Path to RGB JSON file
        target_fps: Target frame rate for synchronization
        max_time_diff: Maximum allowed time difference
    
    Returns:
        List of synchronized frames
    """
    # Load data
    with open(lmc_file, 'r') as f:
        lmc_stream = json.load(f)
    
    with open(rgb_file, 'r') as f:
        rgb_stream = json.load(f)
    
    # Synchronize
    synchronized = synchronize_streams(
        lmc_stream,
        rgb_stream,
        max_time_diff=max_time_diff,
        target_fps=target_fps
    )
    
    return synchronized


def save_synchronized_data(synchronized_frames: List[Dict], output_file: str):
    """
    Save synchronized data to a JSON file.
    
    Args:
        synchronized_frames: List of synchronized frames
        output_file: Output JSON file path
    """
    with open(output_file, 'w') as f:
        json.dump(synchronized_frames, f, indent=2)


def compute_synchronization_stats(synchronized_frames: List[Dict]) -> Dict:
    """
    Compute statistics about synchronization quality.
    
    Args:
        synchronized_frames: List of synchronized frames
    
    Returns:
        Dictionary containing synchronization statistics
    """
    if not synchronized_frames:
        return {
            'num_frames': 0,
            'mean_time_diff': 0.0,
            'max_time_diff': 0.0,
            'std_time_diff': 0.0
        }
    
    time_diffs = [frame['time_diff'] for frame in synchronized_frames]
    
    return {
        'num_frames': len(synchronized_frames),
        'mean_time_diff': float(np.mean(time_diffs)),
        'max_time_diff': float(np.max(time_diffs)),
        'std_time_diff': float(np.std(time_diffs)),
        'median_time_diff': float(np.median(time_diffs))
    }
