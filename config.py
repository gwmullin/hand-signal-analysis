# RTSP Stream URL
RTSP_URL = "rtsp://admin:password@192.168.1.100:554/stream"

# Camera Input Resolution (Must match what the pipe expects or be resized)
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Web Server Configuration
HOST_IP = "0.0.0.0"
HOST_PORT = 5000

# Model Paths
MODEL_DIR = "models"
HAND_DETECTOR_MODEL = f"{MODEL_DIR}/hand_detector.rknn"
HAND_LANDMARK_MODEL = f"{MODEL_DIR}/hand_landmarks_detector.rknn"
GESTURE_EMBEDDER_MODEL = f"{MODEL_DIR}/gesture_embedder.rknn"
GESTURE_CLASSIFIER_MODEL = f"{MODEL_DIR}/canned_gesture_classifier.rknn"
