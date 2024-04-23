from typing import List
from queue import Queue
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import asyncio
import uvicorn

app = FastAPI()


class ConnectionManager:
    def __init__(self):
        # 存放激活的ws连接对象
        self.active_connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        # 等待连接
        await ws.accept()
        # 存储ws连接对象
        self.active_connections.append(ws)

    def disconnect(self, ws: WebSocket):
        # 关闭时 移除ws对象
        self.active_connections.remove(ws)

    @staticmethod
    async def send_personal_message(message: str, ws: WebSocket):
        # 发送个人消息
        await ws.send_text(message)

    async def broadcast(self, message: str):
        # 广播消息
        for connection in self.active_connections:
            await connection.send_text(message)


async def Listen(queue: Queue):
    n = 0
    while True:
        await asyncio.sleep(2)
        n += 1
        queue.put(n)
        print(f'Queue: {queue.qsize()}')

manager = ConnectionManager()
data_queue = Queue()

@app.websocket("/ws/{user}")
async def websocket_endpoint(websocket: WebSocket, user: str):
    await manager.connect(websocket)
    await manager.broadcast(f"用户{user}进入聊天室")

    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"你说了: {data}", websocket)
            await manager.broadcast(f"用户:{user} 说: {data}")
            if not data_queue.empty():
                temp_data = data_queue.get()
                await manager.broadcast(f'广播: {temp_data}')

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"用户-{user}-离开")


if __name__ == "__main__":
    listen_thread = threading.Thread(target=asyncio.run, args=(Listen(data_queue),))
    listen_thread.start()

    uvicorn.run(app, host="127.0.0.1", port=8010)