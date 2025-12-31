import cv2
import numpy as np


# --- CONFIGURATION ---
# COCO Dataset Classes (We only care about a few)
ALL_CLASSES = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 
    7: 'truck', 9: 'traffic light', 11: 'stop sign'
}
TARGET_CLASSES = set(ALL_CLASSES.keys()) # Filter to detect only these

class KivyCamera(Image):
    def __init__(self, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        
        # 1. Load the ONNX model using OpenCV DNN
        # This is much lighter than loading PyTorch
        self.net = cv2.dnn.readNetFromONNX('yolo11n.onnx')
        
        # 2. Setup Camera (0 is usually back camera on generic android/PC)
        self.capture = cv2.VideoCapture(0)
        
        # Scheduling the update loop (30 FPS)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # 3. Preprocess Frame for YOLO
        # YOLOv8 expects 640x640 input, normalized to [0, 1]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # 4. Run Inference
        outputs = self.net.forward()
        
        # YOLOv8 Output shape is usually (1, 84, 8400)
        # Rows: 84 (4 box coords + 80 class probabilities)
        # Cols: 8400 (number of anchors/boxes)
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        confidences = []
        class_ids = []

        # 5. Parse Detections
        # The output matrix is large, so we iterate to find high-confidence boxes
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            
            if maxScore >= 0.25: # Confidence threshold
                # Check if it's a class we care about
                if maxClassIndex in TARGET_CLASSES:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]), # x
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]), # y
                        outputs[0][i][2], # width
                        outputs[0][i][3] # height
                    ]
                    boxes.append(box)
                    confidences.append(float(maxScore))
                    class_ids.append(maxClassIndex)

        # 6. Apply Non-Maximum Suppression (NMS)
        # This removes overlapping boxes for the same object
        result_boxes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

        # 7. Draw Boxes
        if len(result_boxes) > 0:
            for i in result_boxes.flatten():
                x, y, w, h = boxes[i]
                
                # Scale boxes back to original frame size
                x_scale = frame.shape[1] / 640
                y_scale = frame.shape[0] / 640
                
                left = int(x * x_scale)
                top = int(y * y_scale)
                width = int(w * x_scale)
                height = int(h * y_scale)

                label = f"{ALL_CLASSES[class_ids[i]]}: {confidences[i]:.2f}"
                
                # Draw Rectangle
                cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
                # Draw Label
                cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 8. Convert to Kivy Texture for Display
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.texture = texture

    def on_stop(self):
        self.capture.release()

class TrafficCamApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        cam = KivyCamera()
        layout.add_widget(cam)
        return layout

if __name__ == '__main__':
    TrafficCamApp().run()