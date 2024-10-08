import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response, Request, FastAPI, UploadFile, File
from fastapi.responses import PlainTextResponse, HTMLResponse
import io
import cv2
import numpy as np
from PIL import Image

proto_path = r"D:\Code\ML\Model\ncnn\deploy.prototxt"
model_path = r"D:\Code\ML\Model\ncnn\res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

def get_face(img):
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # 遍历检测到的人脸
    face_num = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            face_num += 1
    return face_num



@app.get('/index.html', response_class=HTMLResponse)
async def list_url():
    with open("./index.html", 'r', encoding='utf-8') as file:
        content = file.read()  # 读取文件全部内容
    return content


@app.post("/detect_face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()
    img_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)

    face_num = get_face(img)
    print(f"有 {face_num} 个人脸")


    return {"has_face": f"有 {face_num} 个人脸"}


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
