import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from faster_whisper import WhisperModel
import numpy as np
from zhconv import convert
from pydub import AudioSegment
import threading
import os
import io
import queue

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = FastAPI()

# 加载 Whisper 模型
model = WhisperModel("medium", device="cuda", compute_type="float16")


# 管理客户端连接和音频队列
async def send_message(message: str, websocket: WebSocket):
    await websocket.send_text(message)


class ConnectionManager:
    def __init__(self):
        self.active_connections = []
        self.audio_queue = queue.Queue()
        self.transcription_thread = threading.Thread(target=self.transcribe_loop)
        self.transcription_thread.start()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    def transcribe_loop(self):
        audio_chunks = []
        while True:
            try:
                audio_data = self.audio_queue.get(block=True)
                audio_chunks.append(audio_data)
                print('放入音频')

                # 当收集到足够的音频数据时，执行语音识别
                print("chunk_length: ", len(audio_chunks))
                if len(audio_chunks) >= 3:  # 例如每收到 10 个片段就识别一次
                    audio_data = np.concatenate(audio_chunks)

                    save_audio_to_mp3(audio_data, 'temp.mp3')
                    result = model.transcribe(audio_data)

                    segments, info = result
                    book = ''
                    for segment in segments:
                        book += convert(segment.text, 'zh-cn')
                        print(segment)

                    # 将识别结果广播给所有客户端
                    self.broadcast(book)
                    audio_chunks = []

            except Exception as e:
                print(f"Error: {e}")


def save_audio_to_mp3(audio_data, filename):
    audio_data = (audio_data * 32768).astype(np.int16)
    # 将 NumPy 数组转换为 PyAudio 格式

    # 使用 pydub 从字节数据创建 AudioSegment 对象
    audio_segment = AudioSegment(
        data=audio_data.tobytes(),
        sample_width=2,
        frame_rate=16000,
        channels=1
    )

    # 导出为 MP3 文件
    audio_segment.export(filename, format="mp3")


manager = ConnectionManager()


# WebSocket 路由，用于实时接收音频流
@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    audio_buffer = bytearray()  # 创建一个字节数组用于缓存音频数据

    try:
        while True:
            # 接收音频数据
            data = await websocket.receive_bytes()
            audio_buffer.extend(data)  # 将新的数据追加到缓存中

            audio_length = len(audio_buffer)
            print("接收 -- ", audio_length)

            # 当缓存的数据足够大时,进行处理
            if audio_length >= (16000 * 2):  # 每秒 16000 采样率，2 字节 (int16)
                # 确保缓冲区长度是 2 的倍数
                if len(audio_buffer) % 2 != 0:
                    audio_buffer = audio_buffer[:-1]

                audio_data = np.frombuffer(audio_buffer, dtype=np.int16) / 32768

                audio_buffer = bytearray()  # 清空缓存

                # 将音频数据加入队列
                manager.audio_queue.put(audio_data)

    except WebSocketDisconnect:
        manager.disconnect(websocket)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8888)
