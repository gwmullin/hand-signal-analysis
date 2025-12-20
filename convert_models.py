import os
import zipfile
import shutil
from rknn.api import RKNN

# Constants
TASK_FILE = 'gesture_recognizer.task'
MODEL_DIR = 'models'
TARGET_PLATFORM = 'rk3588'

def recursive_extract(file_path, base_output_dir):
    """Recursively extracts .task and .zip files."""
    try:
        if not zipfile.is_zipfile(file_path):
            return

        # Create a folder for this container if we are deeper in recursion
        # or just extract to base if it's the root file
        current_bad_name = os.path.basename(file_path)
        extract_dir = os.path.join(base_output_dir, current_bad_name + "_extracted")
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            
        print(f"Extracting {file_path} to {extract_dir}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Check specific extraction for nested files
        for root, dirs, files in os.walk(extract_dir):
            for file in files:
                if file.endswith('.task') or file.endswith('.zip'):
                    full_path = os.path.join(root, file)
                    recursive_extract(full_path, extract_dir)
                    
    except Exception as e:
        print(f"Warning: Failed to extract {file_path}: {e}")

def extract_models():
    """Extracts tflite models from the mediapipe .task file"""
    if not os.path.exists(TASK_FILE):
        print(f"Error: {TASK_FILE} not found. Please download it from MediaPipe.")
        return False
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    recursive_extract(TASK_FILE, MODEL_DIR)
    
    print("Extraction complete. Checking for models...")
    # List files to help user identify them if names don't match exactly
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file.endswith('.tflite'):
                print(f"Found: {os.path.join(root, file)}")
    return True

def convert_to_rknn(tflite_path, output_path):
    """Converts a single TFLite model to RKNN"""
    print(f"\nConverting {tflite_path} -> {output_path}")
    
    rknn = RKNN()
    
    # 1. Config
    print("--> Config")
    rknn.config(target_platform=TARGET_PLATFORM)
    
    # 2. Load
    print("--> Loading model")
    ret = rknn.load_tflite(model=tflite_path)
    if ret != 0:
        print("Load failed!")
        return False
        
    # 3. Build
    print("--> Building")
    ret = rknn.build(do_quantization=False) # Start with fp16/no-quant for safety
    if ret != 0:
        print("Build failed!")
        return False
        
    # 4. Export
    print("--> Exporting")
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print("Export failed!")
        return False
        
    print("Done.")
    return True

def main():
    if not extract_models():
        return

    # Map of expected filename -> output rknn name
    # Users might need to adjust keys based on actual extracted names
    models_to_convert = {
        'hand_detector.tflite': 'hand_detector.rknn',
        'hand_landmarks_detector.tflite': 'hand_landmarks_detector.rknn',
        'gesture_embedder.tflite': 'gesture_embedder.rknn',
        'canned_gesture_classifier.tflite': 'canned_gesture_classifier.rknn'
    }

    # Search for these files recursively in MODEL_DIR because zip structure varies
    found_models = {}
    for root, dirs, files in os.walk(MODEL_DIR):
        for file in files:
            if file in models_to_convert:
                found_models[file] = os.path.join(root, file)

    for tflite_name, rknn_name in models_to_convert.items():
        if tflite_name in found_models:
            tflite_path = found_models[tflite_name]
            rknn_path = os.path.join(MODEL_DIR, rknn_name)
            convert_to_rknn(tflite_path, rknn_path)
        else:
            print(f"WARNING: Could not find {tflite_name} in extracted files.")

if __name__ == "__main__":
    main()
