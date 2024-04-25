import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import threading
import uvicorn

app = FastAPI()


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
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


def generate_frames():
    while True:
        frame = video_stream_handler.get_frame()
        if frame is None:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
