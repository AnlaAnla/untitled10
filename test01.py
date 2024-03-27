import cv2

cap = cv2.VideoCapture(0)

if cap.isOpened():
    print('num:', cap.get(cv2.CAP_PROP_FRAME_COUNT))