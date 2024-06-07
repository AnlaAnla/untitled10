import cv2
from ultralytics import YOLO
import numpy as np
import time


def judge_clear(card_img):
    gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    # 计算拉普拉斯算子的方差, 判断图像的模糊程度
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f'卡片图像 拉普拉斯算子的方差: {fm}', end=' \t')
    if fm > 600:
        print(" --------------- 图片清晰")
        return True
    else:
        print("XXXXXX 图片模糊")
        return False


model = YOLO(r"C:\Code\ML\Model\onnx\yolov10_card_4mark_01.onnx", task='detect')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    time.sleep(0.06)

    if success:
        results = model.predict(frame, verbose=False)
        # results = model.predict(frame)
        annotated_frame = results[0].plot()

        cv2.imshow("show", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
