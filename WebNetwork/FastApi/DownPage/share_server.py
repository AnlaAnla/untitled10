from fastapi import FastAPI, File, UploadFile
from pathlib import Path

app = FastAPI()

# 指定要共享的文件所在目录
share_dir = Path(r"D:\Code\ML\Project\untitled10\Nlp\Classify\sentence_judge_bert03")


@app.get("/files")
def list_files():
    files = [f.name for f in share_dir.iterdir() if f.is_file()]
    return {"files": files}


@app.get("/files/{file_name}")
def download_file(file_name: str):
    file_path = share_dir / file_name
    if file_path.exists() and file_path.is_file():
        return {"file_path": str(file_path)}
    else:
        return {"error": "File not found"}


@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    file_path = share_dir / file.filename
    with file_path.open("wb") as f:
        f.write(file.file.read())
    return {"file_path": str(file_path)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8888)
