from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2

app = FastAPI()

@app.websocket("/video_feed")
async def video_feed(websocket: WebSocket):
    await websocket.accept()
    try:
        # 打开视频/摄像头
        cap = cv2.VideoCapture(0)  # 0表示默认摄像头,也可以是视频文件路径
        while True:
            # 读取帧
            ret, frame = cap.read()
            if not ret:
                break
            # 将帧编码为jpeg格式
            _, buffer = cv2.imencode('.jpg', frame)
            # 通过WebSocket发送帧
            await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        # 客户端断开连接,关闭视频/摄像头
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)