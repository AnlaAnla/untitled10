import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import threading
import uvicorn
import asyncio
import time

app = FastAPI()
keep_streaming = True  # 全局变量控制推流循环
video_num = 0


class VideoStreamHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # 打开默认摄像头
        self.frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.start()

    def update_frame(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            with self.lock:
                self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


video_stream_handler = VideoStreamHandler()


@app.get('/video_feed')
async def video_feed():
    global keep_streaming, video_num
    video_num += 1
    if video_num > 1:
        video_num = 0
        await video_stop()

    print('video_num:', video_num)
    keep_streaming = True
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


@app.get('/video_stop')
async def video_stop():
    print('进入video_stop')
    global keep_streaming
    keep_streaming = False
    await asyncio.sleep(0.2)
    return {'status': 'ok'}


def generate_frames():
    global keep_streaming, video_num
    print('开始推流')
    while keep_streaming:
        frame = video_stream_handler.get_frame()
        if frame is None:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转化为灰度图
        ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)  # 阈值二值化

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)
    print('结束推流')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
