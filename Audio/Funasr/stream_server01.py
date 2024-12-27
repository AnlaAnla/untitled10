from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from funasr import AutoModel
import numpy as np
import uvicorn

app = FastAPI()

# 加载模型
chunk_size = [0, 10, 5]  # 每块 600ms
encoder_chunk_look_back = 4
decoder_chunk_look_back = 1
model = AutoModel(model="paraformer-zh-streaming")


@app.websocket("/ws/audio_stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cache = {}  # 缓存初始化
    audio_buffer = []  # 用于存储收到的音频块

    try:
        while True:
            # 接收来自客户端的音频数据
            data = await websocket.receive_bytes()
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer.append(audio_chunk)

            # 拼接音频块并分块处理
            speech = np.concatenate(audio_buffer)
            print("len speech", len(speech))
            chunk_stride = chunk_size[1] * 960  # 根据采样率计算块大小
            while len(speech) >= chunk_stride:
                speech_chunk = speech[:chunk_stride]
                speech = speech[chunk_stride:]  # 剩余音频
                is_final = False

                # 调用模型进行语音识别
                result = model.generate(
                    input=speech_chunk,
                    cache=cache,
                    is_final=is_final,
                    chunk_size=chunk_size,
                    encoder_chunk_look_back=encoder_chunk_look_back,
                    decoder_chunk_look_back=decoder_chunk_look_back,
                )
                print(result)
                await websocket.send_text(result["text"])  # 返回识别结果

            # 更新缓冲区
            audio_buffer = [speech]
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=2345)
