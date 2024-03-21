import cv2
from ultralytics import YOLO
import numpy as np
import time


def get_object(img, boxes):
    if len(boxes) == 0:
        return img
    max_area = 0

    # 选出最大的框
    x1, y1, x2, y2 = 0, 0, 0, 0
    for box in boxes:
        temp_x1, temp_y1, temp_x2, temp_y2 = box
        area = (temp_x2 - temp_x1) * (temp_y2 - temp_y1)
        if area > max_area:
            max_area = area
            x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2

    x1 = int(x1)
    x2 = int(x2)
    y1 = int(y1)
    y2 = int(y2)
    max_img = img[y1:y2, x1:x2, :]

    # max_img = img.crop((x1, y1, x2, y2))
    return max_img


def check_one_card(results):
    result_lenght = len(results[0])
    if result_lenght > 1 or result_lenght == 0:
        # print("检测到的卡片数量大于 1 或等于 0, 请调整图像")
        return False

    # print('此时只有一个卡片')
    return True


def check_new_id(results):
    global card_temp_id
    try:
        card_id = int(results[0].boxes.id.item())
        if card_id in card_temp_id:
            return False
        else:
            card_temp_id.append(card_id)
            return True
    except:
        print("没有检测到新id, 跳过")
        return False


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


model = YOLO('C:\Code\ML\RemoteProject\yolov8_test\model\yolo_card02.pt')
cap = cv2.VideoCapture(0)
card_temp_id = []

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    time.sleep(0.1)

    if success:
        results = model.track(frame, persist=True, verbose=False)
        # results = model.predict(frame)
        annotated_frame = results[0].plot()

        # 仅当有一个卡片并且 id变化时, 记录
        # if check_one_card(results) and check_new_id(results):
        if check_one_card(results):

            conf = results[0].boxes.conf[0].item()
            card_img = get_object(frame, results[0].boxes.xyxy.cpu())

            if conf > 0.75 and judge_clear(card_img):
                print("==========================清晰且可信, 发送图片")
                cv2.imshow("Card YES !!", annotated_frame)

        cv2.imshow("Card NO !!!", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
