from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import uvicorn
import json
import time


app = FastAPI()


@app.post("/train/params_json")
async def upload_json(file: UploadFile = File(...)):
    contents = await file.read()
    json_data = contents.decode("utf-8")
    # 处理JSON数据
    print(json_data)
    json_data = json.loads(json_data)

    print('post:', init_num)
    time.sleep(10)
    print('sleep end')
    return {"message": "JSON file received successfully"}

if __name__ == '__main__':

    print(1111)
    init_num = 3333
    print('main:', init_num)
    uvicorn.run(app, host='127.0.0.1', port=8000)
    print('main:', init_num)

