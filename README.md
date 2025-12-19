# Real-time Gesture Recognition on RK3588 (NPU Accelerated)

This project implements a real-time Hand Gesture Recognition application designed specifically for the Rockchip RK3588 platform. It leverages the onboard NPU (Neural Processing Unit) for high-performance inference and provides a web-based interface to view the live processing feed.

## Architecture

The application is built on a split architecture due to the specific requirements of the Rockchip NPU SDK (RKNPU2).

```mermaid
flowchart LR
    subgraph Host PC [Development PC (x86)]
        A[Mediapipe .task Bundle] -->|Extract| B(4x TFLite Models)
        B -->|rknn-toolkit2| C(4x RKNN Models)
    end

    subgraph RK3588 [Target Board (RK3588)]
        CAM[RTSP Camera] -->|OpenCV| FRAME[Frame Grab]
        FRAME -->|Pre-process| DET[Hand Detection (NPU)]
        DET -->|ROI Crop| LAND[Landmark Detection (NPU)]
        LAND -->|Keypoints| EMB[Gesture Embedding (NPU)]
        EMB -->|Vector| CLASS[Gesture Classifier (NPU)]
        CLASS -->|Gesture ID| DRAW[Annotate Frame]
        DRAW -->|MJPEG| WEB[Flask Web Server]
    end
```

### Why this complexity?
Standard Google MediaPipe libraries run on CPU/GPU and do not support the proprietary Rockchip NPU directly. To achieve real-time performance on embedded hardware, we must:
1.  **Deconstruct** the MediaPipe `gesture_recognizer.task` bundle into its primitive sub-models.
2.  **Convert** these models to the proprietary `.rknn` format.
3.  **Manually Reconstruct** the processing pipeline in Python on the board.

## Prerequisites

### Hardware
*   Board based on Rockchip RK3588 (e.g., Orange Pi 5, Rock 5B).
*   RTSP Camera feed.

### Software (On Board)
1.  **System Dependencies**:
    ```bash
    sudo apt update
    sudo apt install -y python3-dev python3-pip python3-venv git
    ```

2.  **Virtual Environment Setup**:
    It is recommended to use a virtual environment to manage dependencies.
    ```bash
    # Create a virtual environment named 'venv'
    python3 -m venv venv

    # Activate the environment
    source venv/bin/activate
    ```

3.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install RKNN Toolkit Lite2**:
    *Note: This package is not available on PyPI and must be installed from the Rockchip SDK.*
    - Download the appropriate `.whl` file for your Python version (e.g., `rknn_toolkit_lite2-*-cp310-cp310-linux_aarch64.whl`).
    - Install it:
      ```bash
      pip install rknn_toolkit_lite2-*.whl
      ```
*   **rknn-toolkit2**: Required for model conversion (cannot be installed on ARM board).

## Implementation Steps

### 1. Model Preparation (Host PC)
You must first download the `gesture_recognizer.task` from MediaPipe and use a helper script to extract and convert the models.

**Required Models:**
1.  `hand_detector.rknn`
2.  `hand_landmarks_detector.rknn`
3.  `gesture_embedder.rknn`
4.  `canned_gesture_classifier.rknn`

*Note: A `convert_models.py` script will be provided to automate this extraction and conversion using `rknn-toolkit2`.*

### 2. Application Logic (RK3588)

The core application `app.py` will consist of:

#### The `GesturePipeline` Class
This class manages the 4 asynchronous NPU sessions.
*   **Input**: RGB Frame.
*   **Stage 1**: Resize to 224x224 -> Run `HandDetector` -> Get Bounding Box.
*   **Stage 2**: Crop Hand ROI -> Run `HandLandmarkDetector` -> Get 21 Skeleton Points.
*   **Stage 3**: Flatten Keypoints -> Run `GestureEmbedder` -> Get Embedding Vector.
*   **Stage 4**: Input Vector -> Run `GestureClassifier` -> Get Gesture Name (e.g., "Thumbs Up", "Open Palm").

#### The Web Server
A lightweight Flask app is used to stream the results to any browser on the local network.

```python
from flask import Flask, Response
import cv2

app = Flask(__name__)

@app.route('/video_feed')
def video_feed():
    # Returns a multipart MJPEG stream response
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
```

## Running the Application

1.  **Transfer Models**: Copy the 4 `.rknn` files from your PC to the RK3588 board.
2.  **Configure**: Edit `config.py` to set your RTSP URL.
3.  **Run**:
    ```bash
    python3 main.py
    ```
4.  **View**: Open `http://<board-ip>:5000` in your browser.

## References
*   [MediaPipe Gesture Recognition Task](https://developers.google.com/mediapipe/solutions/vision/gesture_recognizer)
*   [ZediOT Blog: MediaPipe via RKNN](https://zediot.com/blog/mediapipe-gesture-recognition-rknn-rk3566/)
*   [MMCC-XX Gesture Sensor](https://github.com/mmcc-xx/gesturesensor)
