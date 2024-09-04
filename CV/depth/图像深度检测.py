import numpy as np
from transformers import pipeline
from PIL import Image
import cv2

# load pipe
pipe = pipeline(task="depth-estimation", model=r"C:\Code\ML\Model\huggingface\depth-anything-small-hf")


def show(img_path):
    img = Image.open(img_path)
    depth = pipe(img)['depth']
    depth.show()


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    img = frame
    if ret:
        frame = Image.fromarray(frame)
        depth = pipe(frame)['depth']
        depth = np.array(depth)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = (img + depth)//2

        # img = np.concatenate((img, depth), axis=0)
        cv2.imshow('frame', depth)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('not ret')

cap.release()
cv2.destroyAllWindows()
