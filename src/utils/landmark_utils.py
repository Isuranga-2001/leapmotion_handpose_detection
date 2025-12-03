"""
Landmark extraction utilities for RGB camera data using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Optional, Tuple, List


class FaceLandmarkExtractor:
    """Extracts facial landmarks and head pose from RGB images using MediaPipe."""
    
    def __init__(self, 
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize MediaPipe Face Mesh.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Key landmark indices for head pose estimation
        # Nose tip, chin, left eye left corner, right eye right corner, left mouth corner, right mouth corner
        self.pose_landmark_indices = [1, 152, 263, 33, 61, 291]
        
    def extract_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """
        Extract facial landmarks from an RGB image.
        
        Args:
            image: RGB image as numpy array (H, W, 3)
        
        Returns:
            Dictionary containing landmarks and pose information, or None if no face detected
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Process the image
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None
        
        # Get the first face
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract landmark coordinates
        h, w = image.shape[:2]
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = landmark.x * w
            y = landmark.y * h
            z = landmark.z * w  # Relative depth
            landmarks.append([x, y, z])
        
        landmarks_array = np.array(landmarks)
        
        # Estimate head pose
        pose = self._estimate_head_pose(landmarks_array, (w, h))
        
        return {
            'landmarks': landmarks_array,  # Shape: (468, 3) or (478, 3) with refinement
            'pose': pose,
            'num_landmarks': len(landmarks_array)
        }
    
    def _estimate_head_pose(self, landmarks: np.ndarray, image_size: Tuple[int, int]) -> Dict[str, float]:
        """
        Estimate head pose (pitch, yaw, roll) from facial landmarks.
        
        Args:
            landmarks: Facial landmark coordinates (N, 3)
            image_size: (width, height) of the image
        
        Returns:
            Dictionary with pitch, yaw, and roll angles in degrees
        """
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = landmarks[self.pose_landmark_indices, :2].astype(np.float64)
        
        # Camera internals
        focal_length = image_size[0]
        center = (image_size[0] / 2, image_size[1] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4, 1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
        
        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Calculate Euler angles
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        
        pitch = euler_angles[0][0]
        yaw = euler_angles[1][0]
        roll = euler_angles[2][0]
        
        return {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll)
        }
    
    def close(self):
        """Release resources."""
        self.face_mesh.close()


def extract_face_landmarks(image: np.ndarray) -> Optional[Dict]:
    """
    Convenience function to extract face landmarks from a single image.
    
    Args:
        image: RGB/BGR image as numpy array
    
    Returns:
        Dictionary containing landmarks and pose, or None if no face detected
    """
    extractor = FaceLandmarkExtractor()
    result = extractor.extract_landmarks(image)
    extractor.close()
    return result


def landmarks_to_features(landmarks_data: Dict) -> np.ndarray:
    """
    Convert landmark data to a compact feature vector.
    
    Args:
        landmarks_data: Dictionary containing 'landmarks' and 'pose' keys
    
    Returns:
        Feature vector combining selected landmarks and pose information
    """
    if landmarks_data is None:
        # Return zero vector if no landmarks detected
        return np.zeros(get_landmark_feature_dimension())
    
    landmarks = landmarks_data['landmarks']  # Shape: (468, 3) or (478, 3)
    pose = landmarks_data['pose']
    
    # Strategy: Use a subset of key landmarks to reduce dimensionality
    # Key facial regions: eyes, nose, mouth, face contour
    key_indices = [
        # Face contour (17 points)
        10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
        # Eyebrows (10 points)
        70, 63, 105, 66, 107, 336, 296, 334, 293, 300,
        # Nose (8 points)
        1, 2, 98, 327, 168, 6, 195, 5,
        # Eyes (12 points)
        33, 133, 160, 159, 158, 157, 173, 263, 362, 385, 386, 387,
        # Mouth (12 points)
        61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308,
        # Chin (3 points)
        152, 377, 148
    ]
    
    # Extract key landmarks
    key_landmarks = landmarks[key_indices].flatten()  # 62 landmarks * 3 = 186 features
    
    # Add pose information
    pose_features = np.array([pose['pitch'], pose['yaw'], pose['roll']])
    
    # Combine features
    feature_vector = np.concatenate([key_landmarks, pose_features])
    
    return feature_vector


def get_landmark_feature_dimension() -> int:
    """
    Get the dimension of the landmark feature vector.
    
    Returns:
        Number of features (62 landmarks * 3 + 3 pose angles = 189)
    """
    return 62 * 3 + 3  # = 189


def visualize_landmarks(image: np.ndarray, landmarks_data: Dict) -> np.ndarray:
    """
    Visualize facial landmarks on the image.
    
    Args:
        image: Input image
        landmarks_data: Dictionary containing landmarks and pose
    
    Returns:
        Image with landmarks drawn
    """
    if landmarks_data is None:
        return image
    
    output_image = image.copy()
    landmarks = landmarks_data['landmarks']
    pose = landmarks_data['pose']
    
    # Draw landmarks
    for landmark in landmarks:
        x, y = int(landmark[0]), int(landmark[1])
        cv2.circle(output_image, (x, y), 1, (0, 255, 0), -1)
    
    # Display pose information
    text = f"Pitch: {pose['pitch']:.1f}, Yaw: {pose['yaw']:.1f}, Roll: {pose['roll']:.1f}"
    cv2.putText(output_image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return output_image
