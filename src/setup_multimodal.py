"""
Quick setup and verification script for multimodal fusion system.
"""

import os
import sys


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_dependencies():
    """Check if required dependencies are installed."""
    print_header("Checking Dependencies")
    
    dependencies = {
        'torch': 'PyTorch',
        'cv2': 'OpenCV',
        'mediapipe': 'MediaPipe',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
    }
    
    missing = []
    
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"✗ {name} NOT installed")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed!")
    return True


def check_leap_motion():
    """Check if Leap Motion SDK is available."""
    print_header("Checking Leap Motion SDK")
    
    # Add paths
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'leapc-python-api', 'src'))
    
    try:
        import leap
        print("✓ Leap Motion SDK found")
        
        # Try to connect
        try:
            connection = leap.Connection()
            connection.connect()
            print("✓ Leap Motion Controller connected")
            connection.disconnect()
            return True
        except Exception as e:
            print(f"✗ Leap Motion Controller NOT connected: {e}")
            print("  Make sure the Leap Motion service is running")
            return False
            
    except ImportError:
        print("✗ Leap Motion SDK not found")
        print("  Install from: leapc-python-api/")
        return False


def check_camera():
    """Check if camera is available."""
    print_header("Checking Camera")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✓ Camera found and accessible")
            cap.release()
            return True
        else:
            print("✗ Camera not accessible")
            print("  Try a different camera ID or check permissions")
            return False
            
    except Exception as e:
        print(f"✗ Camera check failed: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directories")
    
    directories = [
        './data/train',
        './data/val',
        './data/test',
        './data/raw',
        './checkpoints',
        './logs',
        './results'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("\n✓ All directories created!")


def print_next_steps():
    """Print next steps for users."""
    print_header("Next Steps")
    
    print("""
1. Collect Data:
   python src/data_collection/collect_data.py \\
       --gestures "gesture1" "gesture2" "gesture3" \\
       --output-dir ./data/raw \\
       --samples 15

2. Organize data into train/val folders:
   - Move ~80% of samples to ./data/train/
   - Move ~20% of samples to ./data/val/

3. Train Model:
   python src/training/train.py \\
       --train-dir ./data/train \\
       --val-dir ./data/val \\
       --epochs 50 \\
       --batch-size 32

4. Run Inference:
   python src/training/inference.py \\
       --model ./checkpoints/best_model.pth \\
       --labels ./data/labels.json

5. Monitor Training:
   tensorboard --logdir ./logs

📚 For detailed documentation, see: MULTIMODAL_README.md
    """)


def main():
    """Main setup function."""
    print_header("Multimodal Fusion System Setup")
    
    all_checks_passed = True
    
    # Check dependencies
    if not check_dependencies():
        all_checks_passed = False
    
    # Check Leap Motion
    if not check_leap_motion():
        all_checks_passed = False
    
    # Check camera
    if not check_camera():
        all_checks_passed = False
    
    # Create directories
    create_directories()
    
    # Summary
    print_header("Setup Summary")
    
    if all_checks_passed:
        print("✓ Setup complete! System is ready to use.")
        print_next_steps()
    else:
        print("⚠️  Setup incomplete. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Install dependencies: pip install -r requirements.txt")
        print("- Start Leap Motion service")
        print("- Check camera permissions")
        print("- Install Leap SDK: cd leapc-python-api && pip install -e .")


if __name__ == '__main__':
    main()
