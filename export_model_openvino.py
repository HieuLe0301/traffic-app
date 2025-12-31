from ultralytics import YOLO

# Load the standard model
model = YOLO("yolo11n.pt")

# Export it to OpenVINO format
# This creates a folder named 'yolo11n_openvino_model'
model.export(format="openvino", half= True,imgsz=320)