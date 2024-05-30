import cv2
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import threading
import uvicorn
import asyncio

app = FastAPI()


class VideoStreamHandler:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self.update_frame, args=())
        self.thread.start()
        self.client = None  # 记录当前连接
        self.keep_streaming = False  # 控制推流循环

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

    async def gen_frames(self):
        self.keep_streaming = True
        while self.keep_streaming:
            frame = self.get_frame()
            if frame is None:
                continue
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


video_stream_handler = VideoStreamHandler()


@app.get('/video_feed')
async def video_feed(response: Response):
    if video_stream_handler.client is not None:
        video_stream_handler.keep_streaming = False
        await video_stream_handler.client.aclose()

    video_stream_handler.client = response
    try:
        return StreamingResponse(
            video_stream_handler.gen_frames(),
            media_type="multipart/x-mixed-replace;boundary=frame"
        )
    finally:
        video_stream_handler.client = None


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
