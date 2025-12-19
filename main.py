from flask import Flask, render_template, Response
import cv2
import time
import config
from pipeline import GesturePipeline

app = Flask(__name__)

# Global variables
camera = None
pipeline = None

def get_camera():
    global camera
    if camera is None:
        print(f"Connecting to RTSP: {config.RTSP_URL}")
        camera = cv2.VideoCapture(config.RTSP_URL)
        # If RTSP fails, fall back to invalid generic cam or loop
        if not camera.isOpened():
            print("Failed to open RTSP stream.")
            # Optional: Fallback to local webcam 0 for testing
            # camera = cv2.VideoCapture(0)
    return camera

def get_pipeline():
    global pipeline
    if pipeline is None:
        print("Initializing NPU Pipeline...")
        pipeline = GesturePipeline(config)
    return pipeline

def generate_frames():
    cam = get_camera()
    pipe = get_pipeline()
    
    while True:
        success, frame = cam.read()
        if not success:
            print("Read failed, retrying...")
            time.sleep(1)
            # Reconnect logic could go here
            continue
            
        # Run Inference
        processed_frame = pipe.process_frame(frame)
        
        # Encode to JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host=config.HOST_IP, port=config.HOST_PORT, debug=True, threaded=True)
