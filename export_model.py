from ultralytics import YOLO

# 1. This downloads the "brain" (yolo11n.pt) from the internet automatically
print("Downloading model...")
model = YOLO("yolo11n.pt") 

# 2. This converts it into the file you need for your app
print("Converting to ONNX...")
model.export(format="onnx", opset=12)

print("Done! You now have yolo11n.onnx")