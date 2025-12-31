import cv2
from ultralytics import YOLO
import time


def open_camera():
    max_tested = 3
    cam_indices = []

    for i in range(3):
        temp_cam = cv2.VideoCapture(i)
        if temp_cam.isOpened():
            cam_indices.append(i)
            temp_cam.release()

    return cam_indices

# For model running on CUDA, use this 
# model = YOLO('yolo11n.pt')

# For model running on non-CUDA
model = YOLO('yolo11n_openvino_model/', task = 'detect')

#This is for switching cams, for now use index = 1, the first external cam
# cam_indices = open_camera()
# cam_index = cam_indices[0]
# cam = cv2.VideoCapture(cam_index)

cam = cv2.VideoCapture(1)


if not cam.isOpened():
    print("Can't open camera")


tic = time.time()

while True:
    success, frame = cam.read()
    if not success:
        print("Failed to read frame")
        break
    toc = time.time()
    fps = 1 / (toc - tic)
    tic = toc
    
    results = model(frame,stream = True, verbose = False,imgsz = 320)

    person_count = 0
    chair_count = 0
    for result in results:
        annotated_frame = result.plot()
        for box in result.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:
                person_count += 1
            elif class_id == 56:
                chair_count +=1

    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(annotated_frame, f"Persons: {person_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(annotated_frame, f"Chairs: {chair_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow("YOLOv11 Desktop App", annotated_frame)

    key = cv2.waitKey(1) & 0xFF 
    
    if key == ord('q'):
        break
    # elif key == ord('s'):
    #     cam.release()
    #     if not cam_index == len(cam_indices) - 1:
    #         cam_index += 1
    #     else: 
    #         cam_index = 0
    #     cam = cv2.VideoCapture(cam_index)
    
    if cv2.getWindowProperty("YOLOv11 Desktop App", cv2.WND_PROP_VISIBLE) < 1:
        break
        
cv2.destroyAllWindows()