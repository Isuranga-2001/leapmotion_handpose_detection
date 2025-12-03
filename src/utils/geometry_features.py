"""
Geometry feature extraction utilities for Leap Motion hand data.
Computes distances, angles, and palm-related features.
"""

import numpy as np
from typing import List, Dict, Tuple


def compute_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute Euclidean distance between two 3D points."""
    return np.linalg.norm(point1 - point2)


def compute_distances(hand_joints: np.ndarray) -> Dict[str, float]:
    """
    Compute key distances between hand joints.
    
    Args:
        hand_joints: Array of shape (27, 3) containing 3D coordinates of hand joints.
                    Order: wrist, thumb (4), index (4), middle (4), ring (4), pinky (4)
    
    Returns:
        Dictionary containing computed distances.
    """
    if hand_joints.shape != (27, 3):
        raise ValueError(f"Expected hand_joints shape (27, 3), got {hand_joints.shape}")
    
    # Joint indices (based on Leap Motion bone structure)
    WRIST = 0
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12
    RING_TIP = 16
    PINKY_TIP = 20
    INDEX_MCP = 5
    PINKY_MCP = 17
    
    distances = {
        # Fingertip to wrist distances
        'thumb_tip_to_wrist': compute_distance(hand_joints[THUMB_TIP], hand_joints[WRIST]),
        'index_tip_to_wrist': compute_distance(hand_joints[INDEX_TIP], hand_joints[WRIST]),
        'middle_tip_to_wrist': compute_distance(hand_joints[MIDDLE_TIP], hand_joints[WRIST]),
        'ring_tip_to_wrist': compute_distance(hand_joints[RING_TIP], hand_joints[WRIST]),
        'pinky_tip_to_wrist': compute_distance(hand_joints[PINKY_TIP], hand_joints[WRIST]),
        
        # Inter-fingertip distances
        'thumb_index_distance': compute_distance(hand_joints[THUMB_TIP], hand_joints[INDEX_TIP]),
        'index_middle_distance': compute_distance(hand_joints[INDEX_TIP], hand_joints[MIDDLE_TIP]),
        'middle_ring_distance': compute_distance(hand_joints[MIDDLE_TIP], hand_joints[RING_TIP]),
        'ring_pinky_distance': compute_distance(hand_joints[RING_TIP], hand_joints[PINKY_TIP]),
        
        # Palm width (approximate)
        'palm_width': compute_distance(hand_joints[INDEX_MCP], hand_joints[PINKY_MCP]),
    }
    
    return distances


def compute_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
    """
    Compute angle at point2 formed by three points.
    
    Args:
        point1, point2, point3: 3D coordinates as numpy arrays
    
    Returns:
        Angle in degrees
    """
    vector1 = point1 - point2
    vector2 = point3 - point2
    
    # Normalize vectors
    vector1_norm = vector1 / (np.linalg.norm(vector1) + 1e-8)
    vector2_norm = vector2 / (np.linalg.norm(vector2) + 1e-8)
    
    # Compute angle
    cos_angle = np.clip(np.dot(vector1_norm, vector2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def compute_angles(hand_joints: np.ndarray) -> Dict[str, float]:
    """
    Compute key angles between hand joints (finger flexion).
    
    Args:
        hand_joints: Array of shape (27, 3) containing 3D coordinates of hand joints.
    
    Returns:
        Dictionary containing computed angles in degrees.
    """
    if hand_joints.shape != (27, 3):
        raise ValueError(f"Expected hand_joints shape (27, 3), got {hand_joints.shape}")
    
    angles = {}
    
    # Finger names and their joint indices
    fingers = {
        'thumb': [1, 2, 3, 4],
        'index': [5, 6, 7, 8],
        'middle': [9, 10, 11, 12],
        'ring': [13, 14, 15, 16],
        'pinky': [17, 18, 19, 20]
    }
    
    # Compute angles at each joint
    for finger_name, indices in fingers.items():
        for i in range(len(indices) - 2):
            angle = compute_angle(
                hand_joints[indices[i]],
                hand_joints[indices[i + 1]],
                hand_joints[indices[i + 2]]
            )
            angles[f'{finger_name}_joint_{i+1}_angle'] = angle
    
    return angles


def compute_palm_features(hand_joints: np.ndarray) -> Dict[str, float]:
    """
    Compute palm-related features including palm center, radius, and normal vector.
    
    Args:
        hand_joints: Array of shape (27, 3) containing 3D coordinates of hand joints.
    
    Returns:
        Dictionary containing palm features.
    """
    if hand_joints.shape != (27, 3):
        raise ValueError(f"Expected hand_joints shape (27, 3), got {hand_joints.shape}")
    
    # Indices for palm calculation
    WRIST = 0
    INDEX_MCP = 5
    MIDDLE_MCP = 9
    RING_MCP = 13
    PINKY_MCP = 17
    
    # Compute palm center (average of wrist and MCPs)
    palm_points = hand_joints[[WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]]
    palm_center = np.mean(palm_points, axis=0)
    
    # Compute palm radius (average distance from palm center to MCPs)
    distances_to_center = [
        compute_distance(palm_center, hand_joints[INDEX_MCP]),
        compute_distance(palm_center, hand_joints[MIDDLE_MCP]),
        compute_distance(palm_center, hand_joints[RING_MCP]),
        compute_distance(palm_center, hand_joints[PINKY_MCP])
    ]
    palm_radius = np.mean(distances_to_center)
    
    # Compute palm normal vector (cross product of two palm vectors)
    vec1 = hand_joints[INDEX_MCP] - hand_joints[WRIST]
    vec2 = hand_joints[PINKY_MCP] - hand_joints[WRIST]
    palm_normal = np.cross(vec1, vec2)
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
    
    # Compute palm orientation angles
    pitch = np.degrees(np.arcsin(palm_normal[1]))  # Rotation around X-axis
    yaw = np.degrees(np.arctan2(palm_normal[0], palm_normal[2]))  # Rotation around Y-axis
    
    return {
        'palm_center_x': palm_center[0],
        'palm_center_y': palm_center[1],
        'palm_center_z': palm_center[2],
        'palm_radius': palm_radius,
        'palm_normal_x': palm_normal[0],
        'palm_normal_y': palm_normal[1],
        'palm_normal_z': palm_normal[2],
        'palm_pitch': pitch,
        'palm_yaw': yaw
    }


def extract_all_geometric_features(hand_joints: np.ndarray) -> np.ndarray:
    """
    Extract all geometric features from hand joints as a single feature vector.
    
    Args:
        hand_joints: Array of shape (27, 3) containing 3D coordinates of hand joints.
    
    Returns:
        Feature vector combining raw joints, distances, angles, and palm features.
    """
    # Flatten raw joint coordinates (81 features)
    raw_joints = hand_joints.flatten()
    
    # Compute geometric features
    distances = compute_distances(hand_joints)
    angles = compute_angles(hand_joints)
    palm_features = compute_palm_features(hand_joints)
    
    # Combine all features into a single vector
    distance_values = np.array(list(distances.values()))
    angle_values = np.array(list(angles.values()))
    palm_values = np.array(list(palm_features.values()))
    
    # Concatenate all features
    feature_vector = np.concatenate([
        raw_joints,
        distance_values,
        angle_values,
        palm_values
    ])
    
    return feature_vector


def get_feature_dimension() -> int:
    """
    Get the total dimension of the geometric feature vector.
    
    Returns:
        Total number of features
    """
    # 81 (raw joints) + 10 (distances) + 15 (angles) + 9 (palm features)
    return 81 + 10 + 15 + 9  # = 115
