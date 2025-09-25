import leap
import math

def extract_hand_features(hand):
    """
    Extracts geometric features from a Leap Motion hand object.
    Useful for static hand gesture classification (e.g., Random Forest).
    """

    features = {}

    # --- Basic hand info ---
    features["hand_type"] = 0 if str(hand.type) == "HandType.Left" else 1  # 0 = left, 1 = right
    palm = hand.palm
    wrist = hand.arm.prev_joint

    # Palm position (relative to wrist, normalized)
    palm_x = palm.position.x - wrist.x
    palm_y = palm.position.y - wrist.y
    palm_z = palm.position.z - wrist.z
    features["palm_x"] = palm_x
    features["palm_y"] = palm_y
    features["palm_z"] = palm_z

    # Palm orientation (normal vector)
    features["palm_normal_x"] = palm.normal.x
    features["palm_normal_y"] = palm.normal.y
    features["palm_normal_z"] = palm.normal.z

    # Grab & pinch strengths
    features["grab_strength"] = hand.grab_strength
    features["pinch_strength"] = hand.pinch_strength

    # --- Finger features ---
    fingertips = {}
    for finger in hand.digits:
        tip = finger.tip.position
        fingertips[finger.type] = (tip.x, tip.y, tip.z)

        # Store normalized relative to palm
        features[f"finger_{finger.type}_x"] = tip.x - palm.position.x
        features[f"finger_{finger.type}_y"] = tip.y - palm.position.y
        features[f"finger_{finger.type}_z"] = tip.z - palm.position.z

    # --- Distances between key fingertips ---
    def euclidean(a, b):
        return math.sqrt(sum((a[i]-b[i])**2 for i in range(3)))

    if len(fingertips) >= 5:
        thumb = fingertips[0]
        index = fingertips[1]
        middle = fingertips[2]
        ring = fingertips[3]
        pinky = fingertips[4]

        features["dist_thumb_index"] = euclidean(thumb, index)
        features["dist_index_middle"] = euclidean(index, middle)
        features["dist_middle_ring"] = euclidean(middle, ring)
        features["dist_ring_pinky"] = euclidean(ring, pinky)
        features["dist_thumb_pinky"] = euclidean(thumb, pinky)

    # --- Finger angles (finger direction vs. palm normal) ---
    for finger in hand.digits:
        direction = finger.direction
        # angle between finger direction and palm normal
        dot = (direction.x * palm.normal.x +
               direction.y * palm.normal.y +
               direction.z * palm.normal.z)
        mag1 = math.sqrt(direction.x**2 + direction.y**2 + direction.z**2)
        mag2 = math.sqrt(palm.normal.x**2 + palm.normal.y**2 + palm.normal.z**2)
        angle = math.acos(dot / (mag1 * mag2)) if mag1*mag2 != 0 else 0
        features[f"angle_finger_{finger.type}"] = angle

    return features
