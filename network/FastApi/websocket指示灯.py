from fastapi import FastAPI, WebSocket
import asyncio
import uvicorn

app = FastAPI()

# 模拟一个不断变化的值
changing_value = 0

# WebSocket连接管理器
connections = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global changing_value
    await websocket.accept()
    connections.append(websocket)

    try:
        while True:
            await asyncio.sleep(1)  # 每秒更新一次
            changing_value += 1
            for connection in connections:
                await connection.send_text(str(changing_value))
    except Exception as e:
        print(e)
        connections.remove(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)