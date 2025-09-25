import leap
import csv
import time
import json
import numpy as np
import cv2
from functions.extract_features import extract_features, MyListener


class VisualizationCanvas:
    def __init__(self):
        self.name = "Feature Extraction Demo - Hand Tracking"
        self.screen_size = [500, 700]
        self.hands_colour = (255, 255, 255)
        self.font_colour = (0, 255, 44)
        self.feature_colour = (255, 100, 100)
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.current_features = None
        
    def get_joint_position(self, bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
        else:
            return None
            
    def render_hands(self, event):
        self.output_image[:, :] = 0
        
        cv2.putText(
            self.output_image,
            "Capture Hand Features - Press 'q' to Quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            self.font_colour,
            2,
        )
        
        if len(event.hands) == 0:
            cv2.putText(
                self.output_image,
                "No hands detected",
                (10, self.screen_size[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.font_colour,
                1,
            )
            return
            
        for i, hand in enumerate(event.hands):
            for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]
                    
                    wrist = self.get_joint_position(hand.arm.next_joint)
                    elbow = self.get_joint_position(hand.arm.prev_joint)
                    if wrist:
                        cv2.circle(self.output_image, wrist, 4, self.hands_colour, -1)
                    if elbow:
                        cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)
                    if wrist and elbow:
                        cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)
                        
                    bone_start = self.get_joint_position(bone.prev_joint)
                    bone_end = self.get_joint_position(bone.next_joint)
                    
                    if bone_start:
                        cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)
                    if bone_end:
                        cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)
                    if bone_start and bone_end:
                        cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)
                        
                    if ((index_digit == 0) and (index_bone == 0)) or (
                        (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                    ):
                        index_digit_next = index_digit + 1
                        digit_next = hand.digits[index_digit_next]
                        bone_next = digit_next.bones[index_bone]
                        bone_next_start = self.get_joint_position(bone_next.prev_joint)
                        if bone_start and bone_next_start:
                            cv2.line(
                                self.output_image,
                                bone_start,
                                bone_next_start,
                                self.hands_colour,
                                2,
                            )
                            
                    if index_bone == 0 and bone_start and wrist:
                        cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)
                        
            self.display_features_on_canvas(hand, i)
    
    def display_features_on_canvas(self, hand, hand_index):
        if self.current_features:
            y_offset = 60 + (hand_index * 200)
            
            cv2.putText(
                self.output_image,
                f"Hand {hand_index + 1}: {hand.type}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.feature_colour,
                1,
            )
            
            cv2.putText(
                self.output_image,
                f"Grab: {self.current_features.get('grab_strength', 0):.2f}",
                (10, y_offset + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.feature_colour,
                1,
            )
            
            cv2.putText(
                self.output_image,
                f"Pinch: {self.current_features.get('pinch_strength', 0):.2f}",
                (10, y_offset + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.feature_colour,
                1,
            )
            
            palm_pos = f"Palm: ({self.current_features.get('palm_x', 0):.1f}, {self.current_features.get('palm_y', 0):.1f}, {self.current_features.get('palm_z', 0):.1f})"
            cv2.putText(
                self.output_image,
                palm_pos,
                (10, y_offset + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                self.feature_colour,
                1,
            )
    
    def set_current_features(self, features):
        self.current_features = features


class FeatureExtractionDemo:
    
    def __init__(self, canvas=None):
        self.features_collected = []
        self.hand_count = 0
        self.canvas = canvas
        
    def print_feature_details(self, features, hand_type):
        print(f"\n{'='*50}")
        print(f"HAND DETECTED: {hand_type}")
        print(f"{'='*50}")
        
        print("\nPalm Features:")
        print(f"  Position (relative to wrist): ({features['palm_x']:.3f}, {features['palm_y']:.3f}, {features['palm_z']:.3f})")
        print(f"  Normal vector: ({features['palm_normal_x']:.3f}, {features['palm_normal_y']:.3f}, {features['palm_normal_z']:.3f})")
        
        print("\nHand Strength:")
        print(f"  Grab strength: {features['grab_strength']:.3f}")
        print(f"  Pinch strength: {features['pinch_strength']:.3f}")
        
        print("\nFinger Positions (normalized by palm-to-middle-tip distance):")
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        for i, finger_name in enumerate(finger_names):
            x_key = f"finger_{i}_x"
            y_key = f"finger_{i}_y" 
            z_key = f"finger_{i}_z"
            if x_key in features:
                print(f"  {finger_name.capitalize()}: ({features[x_key]:.3f}, {features[y_key]:.3f}, {features[z_key]:.3f})")
        
        print(f"\nTotal features extracted: {len(features)}")
        
    def analyze_gesture_patterns(self, features):
        print("\nGesture Analysis:")
        
        if features['grab_strength'] > 0.8:
            print("Potential FIST gesture detected (high grab strength)")
            
        if 'finger_1_y' in features and features['finger_1_y'] > 0.5:
            print("Potential POINTING gesture detected (index finger extended)")
            
        if features['pinch_strength'] > 0.7:
            print("Potential PINCH gesture detected (high pinch strength)")
            
        if abs(features['palm_normal_z']) > 0.8:
            print("Palm facing forward/backward - potential WAVE gesture")


class DemoListener(leap.Listener):
    
    def __init__(self, demo_instance, canvas=None):
        super().__init__()
        self.demo = demo_instance
        self.canvas = canvas
        self.last_print_time = 0
        self.print_interval = 1.0
        
    def on_connection_event(self, event):
        print("Connected to Leap Motion device!")
        
    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()
        print(f"Found Leap Motion device: {info.serial}")
        
    def on_tracking_event(self, event):
        current_time = time.time()
        
        if self.canvas:
            self.canvas.render_hands(event)
        
        if current_time - self.last_print_time < self.print_interval:
            return
            
        if event.hands:
            for hand in event.hands:
                features = extract_features(hand)
                
                if self.canvas:
                    self.canvas.set_current_features(features)
                
                features['timestamp'] = current_time
                features['hand_type'] = hand.type
                self.demo.features_collected.append(features)
                self.demo.hand_count += 1
                
                self.demo.print_feature_details(features, hand.type)
                
                self.demo.analyze_gesture_patterns(features)
                
            self.last_print_time = current_time
        else:
            if current_time - self.last_print_time >= self.print_interval:
                print("\nNo hands detected - place your hand over the Leap Motion sensor")
                self.last_print_time = current_time


def save_features_to_file(features_list, filename="demo_features.json"):
    try:
        with open(filename, 'w') as f:
            json.dump(features_list, f, indent=2)
        print(f"\nFeatures saved to {filename}")
    except Exception as e:
        print(f"Error saving features: {e}")


def print_demo_instructions():
    print("\n" + "="*60)
    print("LEAP MOTION FEATURE EXTRACTION DEMO WITH VISUALIZATION")
    print("="*60)
    print("This demo showcases the extract_features() function capabilities:")
    print()
    print("Features extracted:")
    print("  - Palm position (relative to wrist)")
    print("  - Palm normal vector (orientation)")
    print("  - Grab and pinch strength")
    print("  - All 5 fingertip positions (normalized)")
    print()
    print("Instructions:")
    print("  - Place your hand(s) over the Leap Motion sensor")
    print("  - Move your fingers to see different feature values")
    print("  - Try different gestures (fist, point, pinch, wave)")
    print("  - Press 'q' in the visualization window to quit")
    print("  - Press Ctrl+C in terminal to stop and save data")
    print()
    print("Visualization Controls:")
    print("  - Real-time hand skeleton rendering")
    print("  - Feature values displayed on screen")
    print("  - Hand type and gesture recognition")
    print("="*60)


def main():
    print_demo_instructions()
    
    canvas = VisualizationCanvas()
    demo = FeatureExtractionDemo(canvas)
    listener = DemoListener(demo, canvas)
    
    connection = leap.Connection()
    connection.add_listener(listener)
    
    try:
        with connection.open():
            print("\nStarting Leap Motion tracking with visualization...")
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            
            running = True
            while running:
                cv2.imshow(canvas.name, canvas.output_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    running = False
                elif key == 27:  # ESC key
                    running = False
                    
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\nDemo stopped")
        
    finally:
        cv2.destroyAllWindows()
        
        print(f"\nDemo Summary:")
        print(f"  - Total hands detected: {demo.hand_count}")
        print(f"  - Feature samples collected: {len(demo.features_collected)}")
        
        if demo.features_collected:
            save_features_to_file(demo.features_collected)
            
            avg_grab = sum(f['grab_strength'] for f in demo.features_collected) / len(demo.features_collected)
            avg_pinch = sum(f['pinch_strength'] for f in demo.features_collected) / len(demo.features_collected)
            
            print(f"  - Average grab strength: {avg_grab:.3f}")
            print(f"  - Average pinch strength: {avg_pinch:.3f}")
        
if __name__ == "__main__":
    main()