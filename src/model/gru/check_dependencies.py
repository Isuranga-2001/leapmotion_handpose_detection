#!/usr/bin/env python3
"""
Quick import verification script to ensure all dependencies are working correctly.
"""

import sys
import os

def check_import(module_name, description=""):
    """Check if a module can be imported successfully."""
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description} - Error: {e}")
        return False

def main():
    print("="*60)
    print("🔍 Checking GRU Model Dependencies")
    print("="*60)
    
    # Core dependencies
    core_deps = [
        ("numpy", "Numerical computing"),
        ("sklearn", "Machine learning library"),
        ("joblib", "Serialization library"),
        ("matplotlib", "Plotting library"),
        ("tensorflow", "Deep learning framework"),
    ]
    
    print("\n📦 Core Dependencies:")
    core_success = 0
    for module, desc in core_deps:
        if check_import(module, desc):
            core_success += 1
    
    # TensorFlow specific imports
    tf_deps = [
        ("tensorflow.keras", "Keras high-level API"),
        ("tensorflow.keras.models", "Keras model classes"),
        ("tensorflow.keras.layers", "Keras layer classes"),
        ("tensorflow.keras.optimizers", "Keras optimizers"),
        ("tensorflow.keras.callbacks", "Keras callbacks"),
        ("tensorflow.keras.utils", "Keras utilities"),
    ]
    
    print("\n🧠 TensorFlow/Keras Components:")
    tf_success = 0
    for module, desc in tf_deps:
        if check_import(module, desc):
            tf_success += 1
    
    # Local modules
    print("\n📁 Local Modules:")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    local_success = 0
    try:
        import model_train
        print(f"✅ model_train - Main training module")
        local_success += 1
        
        # Check specific functions
        required_functions = [
            'load_gru_data', 
            'preprocess_data', 
            'build_gru_model', 
            'train_gru_model',
            'load_model_artifacts',
            'predict_gesture'
        ]
        
        for func_name in required_functions:
            if hasattr(model_train, func_name):
                print(f"  ✅ {func_name}")
            else:
                print(f"  ❌ {func_name} - Missing function")
        
    except ImportError as e:
        print(f"❌ model_train - Main training module - Error: {e}")
    
    # Summary
    print("\n" + "="*60)
    print("📊 Summary:")
    print(f"  Core Dependencies: {core_success}/{len(core_deps)}")
    print(f"  TensorFlow Components: {tf_success}/{len(tf_deps)}")
    print(f"  Local Modules: {local_success}/1")
    
    total_success = core_success + tf_success + local_success
    total_checks = len(core_deps) + len(tf_deps) + 1
    
    if total_success == total_checks:
        print("🎉 All dependencies are working correctly!")
        print("You can now proceed with training and using the GRU model.")
    else:
        print("⚠️  Some dependencies are missing. Please install them using:")
        print("  pip install -r requirements.txt")
        
    print("="*60)
    
    return total_success == total_checks

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)