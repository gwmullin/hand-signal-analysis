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
        # Normalization: MediaPipe TFLite models usually expect [0, 1] float32
        img = img.astype(np.float32) / 255.0
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
        det_input = self.preprocess_image(frame, (224, 224))
        try:
            det_out = self.det.run(det_input)
        except Exception as e:
            print(f"Inference Error (Det): {e}")
            return vis_frame
        
        # Helper: Parse detection output
        # WARNING: This is a simplifed parser. Real production use requires 
        # decoding the specific SSD anchor boxes of the model version used.
        box = self._parse_detection(det_out, frame.shape)
        
        result_text = "No Hand"
        
        if box is not None:
            x, y, w, h = box
            # Draw ROI
            cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # 2. Extract ROI for Landmarks
            # Ensure ROI is within bounds
            h_img, w_img, _ = frame.shape
            y = max(0, y); x = max(0, x)
            h = min(h, h_img - y); w = min(w, w_img - x)
            
            roi = frame[y:y+h, x:x+w]
            if roi.size > 0:
                land_input = self.preprocess_image(roi, (224, 224))
                try:
                    land_out = self.landmark.run(land_input)
                    
                    # 3. Landmarks -> Embedding
                    # Note: gesture_embedder often takes the raw landmark tensor (1, 42) or (1, 63)
                    # We pass the output of landmark detector directly
                    embed_out = self.embedder.run(land_out[0])
                    
                    # 4. Classification
                    # Takes the embedding vector
                    class_out = self.classifier.run(embed_out[0])
                    
                    # Get Class
                    # Usually class_out is [probability_vector]
                    probs = class_out[0][0] # Adjust indexing based on actual shape (1, N)
                    idx = np.argmax(probs)
                    score = probs[idx]
                    
                    if score > 0.5:
                        result_text = f"{self.labels[idx]} ({score:.2f})"
                    else:
                        result_text = "Uncertain"
                        
                except Exception as e:
                    # In case of shape mismatches during development
                    print(f"Pipeline flow error: {e}")
                    result_text = "Pipeline Error"

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
