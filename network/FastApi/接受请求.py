from fastapi import FastAPI, File, UploadFile, Request
from typing import List
import asyncio
import uvicorn
import json
import time
import pandas as pd
import os

os.path.splitext()

app = FastAPI()


@app.get("/test/{front_or_back}")
async def test(front_or_back: int):
    print(front_or_back)
    return {'yes!!'}

@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    content = await file.read()
    with open('temp.csv', 'wb') as f:
        f.write(content)


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

