import cv2
import numpy as np
try:
    from rknn.lite.api import RKNNLite
except ImportError:
    print("Warning: rknn-toolkit-lite2 not found. NPU inference will fail.")
    # Dummy class for development/linting if library is missing
    class RKNNLite:
        def load_rknn(self, path): pass
        def init_runtime(self, core_mask=None): pass
        def inference(self, inputs): return []
        def release(self): pass

class RKNNSession:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        print(f"Loading {model_path}...")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print(f"Load failed for {model_path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            print(f"Init runtime failed for {model_path}")

    def run(self, input_data):
        # input_data should be [np.array]
        return self.rknn.inference(inputs=[input_data])
    
    def release(self):
        self.rknn.release()

class GesturePipeline:
    def __init__(self, config):
        self.det = RKNNSession(config.HAND_DETECTOR_MODEL)
        self.landmark = RKNNSession(config.HAND_LANDMARK_MODEL)
        self.embedder = RKNNSession(config.GESTURE_EMBEDDER_MODEL)
        self.classifier = RKNNSession(config.GESTURE_CLASSIFIER_MODEL)
        
        # Labels from MediaPipe gesture_classifier
        self.labels = ['None', 'Closed_Fist', 'Open_Palm', 'Pointing_Up', 'Thumb_Down', 'Thumb_Up', 'Victory', 'ILoveYou']

    def preprocess_image(self, img, target_size=(224, 224)):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        # Add batch dimension: (1, 224, 224, 3)
        input_data = np.expand_dims(img, axis=0)
        return input_data

    def process_frame(self, frame):
        """
        Main pipeline: Frame -> Detect -> Landmark -> Embed -> Classify
        Returns: (processed_frame, result_dict)
        """
        # copy frame for drawing
        vis_frame = frame.copy()
        
        # 1. Hand Detection
        # Note: Actual model input size might differ, usually 224x224 or 192x192 depending on model version
        det_input = self.preprocess_image(frame, (224, 224))
        det_out = self.det.run(det_input)
        
        # Helper: Parse detection output (Simplified logic)
        # In a real scenario, you need to parse the SSD anchors / bounding boxes
        # For this example, we'll assume we get a box or None.
        # This part requires specific knowledge of the output tensor shape of hand_detector.tflite
        box = self._parse_detection(det_out, frame.shape)
        
        result_text = "No Hand"
        
        if box is not None:
            x, y, w, h = box
            # Draw ROI
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 2. Extract ROI for Landmarks
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                land_input = self.preprocess_image(roi, (224, 224))
                land_out = self.landmark.run(land_input)
                
                # 3. Landmarks -> Embedding
                # Flatten landmarks (21 * 3) or similar structure
                # This is highly dependent on model specifics
                # Assuming land_out produces the correct shape for embedder
                
                # For demonstration, we pass dummy data if shapes mismatch
                # In production, you verify shapes: print(land_out[0].shape)
                
                # 4. Classification
                # Passing the raw detection/landmark output is not enough; 
                # the embedder needs a specific vector derived from landmarks.
                # Since exact tensor parsing is complex, we'll assume a dummy flow:
                
                # Placeholder for valid integration:
                # embed_out = self.embedder.run(land_out[0]) 
                # class_out = self.classifier.run(embed_out[0])
                
                # Using a dummy result for now to ensure code structure works
                # idx = np.argmax(class_out[0])
                idx = 2 # Fake "Open_Palm"
                
                result_text = self.labels[idx] if idx < len(self.labels) else "Unknown"

        cv2.putText(vis_frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return vis_frame

    def _parse_detection(self, output, shape):
        # TODO: Implement actual SSD output parsing (scores, boxes)
        # This usually involves sigmoid, decoding generic SSD box coords, NMS.
        # Returning a dummy box for the center of the screen for testing flow
        h, w, _ = shape
        return (int(w*0.3), int(h*0.3), int(w*0.4), int(h*0.4))

    def release(self):
        self.det.release()
        self.landmark.release()
        self.embedder.release()
        self.classifier.release()
