import leap
import time
import math
import csv

# ---------- Feature Extraction Helper ----------
def extract_features(hand):
    """
    Extracts static pose features from a single hand.
    Features include normalized joint positions, inter-finger distances, angles, and hand attributes.
    Returns a flat feature vector (list of floats).
    """

    features = []

    # --- Palm features ---
    palm_pos = hand.palm.position
    palm_normal = hand.palm.normal
    features += [palm_pos.x, palm_pos.y, palm_pos.z]          # Palm position
    features += [palm_normal.x, palm_normal.y, palm_normal.z] # Palm normal

    # Grab/Pinch strength
    features.append(hand.grab_strength)
    features.append(hand.pinch_strength)
    features.append(hand.palm.width)

    # --- Finger features ---
    # Normalize relative to palm position
    for finger in hand.digits:
        tip = finger.tip.position
        tip_x = tip.x - palm_pos.x
        tip_y = tip.y - palm_pos.y
        tip_z = tip.z - palm_pos.z
        features += [tip_x, tip_y, tip_z]

        # Finger direction (unit vector)
        dir_vec = finger.direction
        features += [dir_vec.x, dir_vec.y, dir_vec.z]

        # Finger length & width
        features.append(finger.length)
        features.append(finger.width)

    # --- Inter-finger distances (tips only) ---
    tips = [f.tip.position for f in hand.digits]
    for i in range(len(tips)):
        for j in range(i+1, len(tips)):
            dx = tips[i].x - tips[j].x
            dy = tips[i].y - tips[j].y
            dz = tips[i].z - tips[j].z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            features.append(dist)

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
