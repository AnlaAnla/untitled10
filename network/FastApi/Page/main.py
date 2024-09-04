import fastapi
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Response, Request, FastAPI
from fastapi.responses import FileResponse
from fastapi.responses import PlainTextResponse, HTMLResponse
import io
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get('/index.html', response_class=HTMLResponse)
async def list_url():
    with open("./index.html", 'r') as file:
        content = file.read()  # 读取文件全部内容
    return content


@app.get("/image")
def get_image():
    print('收到请求')
    img = Image.open(r"C:\Users\wow38\Pictures\scenery\31417482b115ea49824edb27b865faf3.jpg")
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    byte_arr.seek(0)
    return Response(content=byte_arr.getvalue(), media_type="image/png")


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
