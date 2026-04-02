import cv2
import time

# 1. 配置参数
rtsp_url = "rtsp://admin:password@192.168.77.10:554/live/ch0"
output_file = "camera_record.mp4"
record_seconds = 6  # 录制时长（秒）

# 2. 连接摄像头
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("无法连接 RTSP 视频流")
    exit()

# 3. 获取视频流属性（分辨率和帧率）
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: fps = 25  # 如果获取不到 FPS，手动指定一个常用值

# 4. 定义视频编码器和输出对象 (推荐使用 mp4v 或 XVID)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

print(f"正在录制 {record_seconds} 秒视频...")
start_time = time.time()

while (time.time() - start_time) < record_seconds:
    ret, frame = cap.read()
    if ret:
        # 将当前帧写入文件
        out.write(frame)

        # (可选) 在窗口显示预览，按 'q' 键可提前退出
        # cv2.imshow('Recording...', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    else:
        print("丢失帧数据")
        break

# 5. 释放资源
cap.release()
out.release()
print(f"录制完成，文件保存为: {output_file}")
