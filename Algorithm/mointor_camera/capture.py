import cv2

# 替换为你的摄像头真实账号密码
rtsp_url = "rtsp://admin:password@192.168.77.10:554/live/ch0"

cap = cv2.VideoCapture(rtsp_url)

if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # 保存图片
        cv2.imwrite("snapshot.jpg", frame)
        print("照片已保存为 snapshot.jpg")
    cap.release()
else:
    print("无法打开视频流，请检查 RTSP 地址或账号密码")