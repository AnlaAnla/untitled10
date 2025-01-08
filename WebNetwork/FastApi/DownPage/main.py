from fastapi import FastAPI
from fastapi.responses import FileResponse
import os

app = FastAPI()

# 文件路径
FILE_PATH = r"C:\Users\martin\Downloads\212.txt"  # 请替换为你要下载的文件路径


@app.get("/download")
async def download_file():
    # 检查文件是否存在
    if os.path.exists(FILE_PATH):
        return FileResponse(FILE_PATH, media_type='application/octet-stream', filename=os.path.basename(FILE_PATH))
    else:
        return {"error": "File not found"}


# curl -O http://your_server_ip:8000/download/
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
