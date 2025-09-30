import leap
import time
import os
import json
import numpy as np
import cv2
import tkinter as tk
from tkinter import simpledialog
from functions.extract_features import extract_features


class VisualizationCanvas:
    def __init__(self):
        self.name = "GRU Data Capture - Hand Tracking"
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
            
    def render_hands(self, event, capturer=None):
        self.output_image[:, :] = 0
        
        cv2.putText(
            self.output_image,
            "Press 's' to start countdown, 'c' to set class, 'space/x' to stop & save, 'ESC' to quit",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            self.font_colour,
            2,
        )
        
        # Show current class name
        if capturer and capturer.current_class_name:
            cv2.putText(
                self.output_image,
                f"Current Class: {capturer.current_class_name}",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),  # Cyan color for class name
                2,
            )
            y_offset = 70
        else:
            cv2.putText(
                self.output_image,
                "No class set - Press 'c' to set class name",
                (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),  # Red color for warning
                2,
            )
            y_offset = 70
        
        # Show capture status and countdown
        if capturer:
            if capturer.countdown_active:
                countdown_text = capturer.get_countdown_text()
                if countdown_text:
                    cv2.putText(
                        self.output_image,
                        countdown_text,
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),  # Yellow color for countdown
                        3,
                    )
            else:
                status_color = (0, 255, 0) if capturer.is_recording else (0, 0, 255)
                status_text = "RECORDING" if capturer.is_recording else "READY"
                cv2.putText(
                    self.output_image,
                    status_text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    status_color,
                    2,
                )
            
            if capturer.is_recording:
                cv2.putText(
                    self.output_image,
                    f"Frames: {len(capturer.current_sequence)}/{capturer.seq_len}",
                    (10, y_offset + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
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
            # Draw hand skeleton
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
            y_offset = 100 + (hand_index * 200)
            
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


class GRUCapture:
    def __init__(self, save_dir="gru_data", seq_len=50):
        self.seq_len = seq_len
        self.current_sequence = []
        self.sequences = []
        self.save_dir = save_dir
        self.is_recording = False
        self.countdown_active = False
        self.countdown_start_time = 0
        self.countdown_duration = 3  # 3 seconds
        self.current_class_name = None  # Store current class name
        os.makedirs(save_dir, exist_ok=True)

    def set_class_name(self):
        """Ask user for class name and store it"""
        root = tk.Tk()
        root.withdraw()
        class_name = simpledialog.askstring(
            "Class Name Input", 
            "Enter class name for sequences:",
            parent=root
        )
        root.destroy()
        
        if class_name:
            self.current_class_name = class_name
            print(f"[INFO] Class name set to: {class_name}")
        else:
            print("[WARN] No class name entered")
        
        return class_name is not None

    def start_countdown(self):
        """Start the 3-second countdown before recording"""
        if self.is_recording or self.countdown_active:
            print("[WARN] Already recording or countdown in progress")
            return
        
        # Check if class name is set, if not ask for it
        if not self.current_class_name:
            print("[INFO] No class name set. Please enter a class name first.")
            if not self.set_class_name():
                print("[WARN] Cannot start recording without class name")
                return
        
        self.countdown_active = True
        self.countdown_start_time = time.time()
        print(f"\n[INFO] Starting countdown for class '{self.current_class_name}'...")

    def update_countdown(self):
        """Update countdown and start recording when finished"""
        if not self.countdown_active:
            return False
        
        elapsed = time.time() - self.countdown_start_time
        remaining = self.countdown_duration - elapsed
        
        if remaining <= 0:
            # Countdown finished, start recording
            self.countdown_active = False
            self.start_sequence()
            return False
        
        return True

    def get_countdown_text(self):
        """Get current countdown text for display"""
        if not self.countdown_active:
            return None
        
        elapsed = time.time() - self.countdown_start_time
        remaining = self.countdown_duration - elapsed
        countdown_num = int(remaining) + 1
        
        if countdown_num > 0:
            return f"Starting in {countdown_num}..."
        else:
            return "GO!"

    def start_sequence(self):
        self.current_sequence = []
        self.is_recording = True
        print("\n[INFO] Sequence recording started!")

    def stop_and_save(self):
        if not self.is_recording:
            print("[WARN] Not currently recording")
            return
            
        self.is_recording = False
        
        if not self.current_sequence:
            print("[WARN] No sequence recorded")
            return
        
        if not self.current_class_name:
            print("[ERROR] No class name set, cannot save sequence")
            return
        
        # Save as .npy file using current class name
        arr = np.array(self.current_sequence, dtype=np.float32)
        filename = f"{self.current_class_name}_{int(time.time())}.npy"
        filepath = os.path.join(self.save_dir, filename)
        np.save(filepath, arr)

        print(f"[INFO] Sequence saved -> {filepath} (shape={arr.shape})")
        self.sequences.append((filepath, self.current_class_name))

    def add_frame(self, features):
        if self.is_recording:
            vec = list(features.values())
            self.current_sequence.append(vec)
            
            if len(self.current_sequence) >= self.seq_len:
                print(f"[INFO] Sequence reached max length ({self.seq_len} frames), stopping recording.")
                self.stop_and_save()

class GRUListener(leap.Listener):
    def __init__(self, capturer, canvas=None):
        super().__init__()
        self.capturer = capturer
        self.canvas = canvas
        self.last_time = 0
        self.interval = 1.0 / 30  # ~30 FPS

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
        now = time.time()
        
        # Update countdown if active
        if self.capturer.countdown_active:
            self.capturer.update_countdown()
        
        # Always update visualization
        if self.canvas:
            self.canvas.render_hands(event, self.capturer)
        
        # Control frame rate for recording
        if now - self.last_time < self.interval:
            return
        self.last_time = now

        # Process hands for recording (only if actually recording, not during countdown)
        if event.hands and self.capturer.is_recording:
            for hand in event.hands:
                features = extract_features(hand)
                
                # Update canvas with current features
                if self.canvas:
                    self.canvas.set_current_features(features)
                
                # Add timestamp and hand type
                features['timestamp'] = now
                features['hand_type'] = 1 if hand.type == leap.HandType.Right else 0
                
                # Only add to sequence if recording
                self.capturer.add_frame(features)
        else:
            # Show message when no hands detected or during countdown
            if self.canvas and not self.capturer.countdown_active:
                self.canvas.set_current_features(None)


def main():
    canvas = VisualizationCanvas()
    capturer = GRUCapture(save_dir="gru_data", seq_len=50)
    listener = GRUListener(capturer, canvas)

    connection = leap.Connection()
    connection.add_listener(listener)

    try:
        with connection.open():
            print(f"\nStarting Leap Motion tracking for GRU data capture...")
            print(f"Sequence length: {capturer.seq_len} frames")
            print(f"Save directory: {capturer.save_dir}")
            print("\nControls:")
            print("  'c' = Set class name")
            print("  's' = Start 3-second countdown then record")
            print("  'space/x' = Stop & save sequence")
            print("  'ESC' = Quit")
            
            connection.set_tracking_mode(leap.TrackingMode.Desktop)
            
            running = True
            while running:
                cv2.imshow(canvas.name, canvas.output_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    running = False
                elif key == ord('c'):  # 'c' - set class name
                    capturer.set_class_name()
                elif key == ord('s'):  # Start countdown
                    capturer.start_countdown()
                elif key == 32 or key == ord('x'):  # Space or 'x' - stop and save
                    capturer.stop_and_save()
                    
                time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
        
    finally:
        cv2.destroyAllWindows()
        
        print(f"\nSession Summary:")
        print(f"  - Current class: {capturer.current_class_name or 'Not set'}")
        print(f"  - Total sequences captured: {len(capturer.sequences)}")
        
        if capturer.sequences:
            for i, (filepath, class_name) in enumerate(capturer.sequences, 1):
                print(f"  - Sequence {i}: {class_name} -> {os.path.basename(filepath)}")


if __name__ == "__main__":
    main()
