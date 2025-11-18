import leap

# ---------- Feature Extraction Helper ----------
def extract_features(hand):
    """
    Extract Leap Motion hand features with wrist-based normalization.
    Wrist (arm.prev_joint) is set as origin (0,0,0).
    """
    features = {}

    wrist = hand.arm.prev_joint  # wrist joint
    palm = hand.palm.position

    # Normalized palm position
    features["palm_x"] = palm.x - wrist.x
    features["palm_y"] = palm.y - wrist.y
    features["palm_z"] = palm.z - wrist.z

    # Palm normal (orientation)
    features["palm_normal_x"] = hand.palm.normal.x
    features["palm_normal_y"] = hand.palm.normal.y
    features["palm_normal_z"] = hand.palm.normal.z

    # Grab & pinch strengths
    features["grab_strength"] = hand.grab_strength
    features["pinch_strength"] = hand.pinch_strength

    # Reference scale (palm to middle fingertip distance)
    # Get the tip of middle finger (digit 2) using the last bone's next_joint
    middle_digit = hand.digits[2]
    middle_tip = middle_digit.bones[3].next_joint  # Distal bone's next joint is the tip
    ref_dist = ((middle_tip.x - wrist.x) ** 2 +
                (middle_tip.y - wrist.y) ** 2 +
                (middle_tip.z - wrist.z) ** 2) ** 0.5
    ref_dist = ref_dist if ref_dist != 0 else 1.0

    # Finger tip positions (relative to wrist, scaled)
    for finger_index, finger in enumerate(hand.digits):
        # Get fingertip as the next_joint of the distal bone (index 3)
        tip = finger.bones[3].next_joint
        features[f"finger_{finger_index}_x"] = (tip.x - wrist.x) / ref_dist
        features[f"finger_{finger_index}_y"] = (tip.y - wrist.y) / ref_dist
        features[f"finger_{finger_index}_z"] = (tip.z - wrist.z) / ref_dist

    return features

# ---------- Listener Class ----------
class MyListener(leap.Listener):
    def __init__(self, csv_writer=None):
        super().__init__()
        self.csv_writer = csv_writer

    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        for hand in event.hands:
            features = extract_features(hand)
            print(f"Extracted {len(features)} features for {hand.type}")

            # Save to CSV if writer is available
            if self.csv_writer:
                self.csv_writer.writerow(features)


# # ---------- Main ----------
# def main():
#     with open("leap_features.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         listener = MyListener(csv_writer=writer)

#         connection = leap.Connection()
#         connection.add_listener(listener)

#         with connection.open():
#             connection.set_tracking_mode(leap.TrackingMode.Desktop)
#             try:
#                 while True:
#                     time.sleep(0.1)
#             except KeyboardInterrupt:
#                 print("Stopping...")


# if __name__ == "__main__":
#     main()
