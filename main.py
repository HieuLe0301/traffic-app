import cv2
from ultralytics import YOLO


def open_camera():
    max_tested = 3
    cam_indices = []

    for i in range(3):
        temp_cam = cv2.VideoCapture(i)
        if temp_cam.isOpened():
            cam_indices.append(i)
            temp_cam.release()

    return cam_indices


model = YOLO('yolo11n.pt')
cam_indices = open_camera()
cam_index = cam_indices[0]
cam = cv2.VideoCapture(cam_index)


if not cam.isOpened():
    print("Can't open camera")

while True:
    success, frame = cam.read()
    if not success:
        print("Failed to read frame")
        break
    
    results = model(frame,stream = True, verbose = False)

    for result in results:
        annotated_frame = result.plot()
    
    cv2.imshow("YOLOv11 Desktop App", annotated_frame)

    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cam.release()
        if not cam_index == len(cam_indices) - 1:
            cam_index += 1
        else: 
            cam_index = 0
        cam = cv2.VideoCapture(cam_index)
        
        

        


cv2.destroyAllWindows()