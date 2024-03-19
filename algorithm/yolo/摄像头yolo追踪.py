import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('C:\Code\ML\RemoteProject\yolov8_test\model\yolo_card02.pt')
#
# # Open the video file
# video_path = "path/to/video.mp4"
# cap = cv2.VideoCapture(video_path)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        results = model.track(frame, persist=True)

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.temp_id
        # 可能会出现没有检测到物体的情况
        if track_ids is not None:
            track_ids = track_ids.int().cpu().tolist()
            print(track_ids)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # cv2.imshow('camera', frame)
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
