import cv2
from ultralytics import YOLO


model = YOLO('C:\Code\ML\RemoteProject\yolov8_test\model\yolo_card02.pt')

video_path = r"C:\Code\ML\Video\2024_03_07 16_20_42.mp4"
save_path = r"C:\Code\ML\Video\temp\output3.mp4"

cap = cv2.VideoCapture(video_path)

# 获取视频的帧率、宽度和高度
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 创建视频写入对象，指定输出视频的文件名、编码器、帧率和大小
writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

num = 0
while True:
    ret, frame = cap.read()
    if ret:
        # 对图像进行处理
        results = model.track(frame, persist=True)
        frame = results[0].plot()
        # frame = cv2.resize(frame, (width // 2, height // 2))

        writer.write(frame)

        num += 1
        print(num)
    else:
        print('not ret')
        break

cap.release()
writer.release()
