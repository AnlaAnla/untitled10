from __future__ import annotations

import asyncio
import hmac
import os
import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

APP_TITLE = "LAN File Share Space"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "storage"
SHARED_DIR = DATA_DIR / "shared"
INBOX_DIR = DATA_DIR / "inbox"
ADMIN_COOKIE = "fileshare_admin"
ADMIN_KEY = os.getenv("FILESHARE_ADMIN_KEY", "").strip()
MAX_UPLOAD_MB = int(os.getenv("FILESHARE_MAX_UPLOAD_MB", "20480"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
CHUNK_SIZE = 1024 * 1024
SAFE_NAME_RE = re.compile(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+")
ADMIN_SESSIONS: set[str] = set()

for folder in (DATA_DIR, SHARED_DIR, INBOX_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title=APP_TITLE,
    description="A lightweight LAN file sharing service built with FastAPI.",
)


class AdminLoginRequest(BaseModel):
    key: str


class FileTooLargeError(Exception):
    pass


def render_index() -> str:
    index_path = BASE_DIR / "index.html"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as file:
            return file.read()
    else:
        raise HTTPException(status_code=404, detail="index.html not found")


def safe_filename(filename: str | None) -> str:
    raw_name = (filename or "").strip()
    if not raw_name:
        raw_name = f"file_{datetime.now():%Y%m%d_%H%M%S}"
    cleaned = SAFE_NAME_RE.sub("_", Path(raw_name).name)
    cleaned = cleaned.strip("._") or f"file_{datetime.now():%Y%m%d_%H%M%S}"
    return cleaned[:180]


def make_unique_path(directory: Path, original_name: str) -> Path:
    candidate = directory / original_name
    if not candidate.exists():
        return candidate

    suffix = "".join(candidate.suffixes)
    stem = candidate.name[:-len(suffix)] if suffix else candidate.name
    counter = 2
    while True:
        renamed = directory / f"{stem}_{counter}{suffix}"
        if not renamed.exists():
            return renamed
        counter += 1


def file_info(path: Path) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


def list_files(directory: Path) -> list[dict[str, int | str]]:
    files = [path for path in directory.iterdir() if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [file_info(path) for path in files]


def is_admin(request: Request) -> bool:
    if not ADMIN_KEY:
        return True

    direct_key = request.headers.get("X-Admin-Key", "").strip()
    if direct_key and hmac.compare_digest(direct_key, ADMIN_KEY):
        return True

    session_token = request.cookies.get(ADMIN_COOKIE, "")
    return bool(session_token and session_token in ADMIN_SESSIONS)


def require_admin(request: Request) -> None:
    if not is_admin(request):
        raise HTTPException(status_code=401, detail="需要管理权限")


def resolve_space(space: Literal["shared", "inbox"]) -> Path:
    return SHARED_DIR if space == "shared" else INBOX_DIR


def write_upload_file(source, target: Path) -> None:
    written = 0
    with target.open("wb") as buffer:
        while True:
            chunk = source.read(CHUNK_SIZE)
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise FileTooLargeError(f"文件超过限制，单文件最大 {MAX_UPLOAD_MB} MB")
            buffer.write(chunk)


async def save_upload(upload: UploadFile, directory: Path) -> dict[str, int | str]:
    target = make_unique_path(directory, safe_filename(upload.filename))

    try:
        await asyncio.to_thread(write_upload_file, upload.file, target)
    except FileTooLargeError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()

    return file_info(target)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return render_index()


@app.get("/api/files")
async def api_files(request: Request) -> JSONResponse:
    admin = is_admin(request)
    return JSONResponse(
        {
            "admin_required": bool(ADMIN_KEY),
            "is_admin": admin,
            "shared": list_files(SHARED_DIR),
            "inbox": list_files(INBOX_DIR) if admin else [],
        }
    )


@app.post("/api/admin/login")
async def admin_login(payload: AdminLoginRequest) -> JSONResponse:
    if not ADMIN_KEY:
        return JSONResponse({"ok": True, "message": "管理密钥未启用"})
    if not hmac.compare_digest(payload.key.strip(), ADMIN_KEY):
        raise HTTPException(status_code=401, detail="管理密钥错误")

    token = secrets.token_urlsafe(32)
    ADMIN_SESSIONS.add(token)
    response = JSONResponse({"ok": True})
    response.set_cookie(
        key=ADMIN_COOKIE,
        value=token,
        httponly=True,
        samesite="lax",
        secure=False,
        max_age=86400,
    )
    return response


@app.post("/api/admin/logout")
async def admin_logout(request: Request) -> JSONResponse:
    token = request.cookies.get(ADMIN_COOKIE, "")
    if token:
        ADMIN_SESSIONS.discard(token)
    response = JSONResponse({"ok": True})
    response.delete_cookie(ADMIN_COOKIE)
    return response


@app.post("/api/upload/shared")
async def upload_shared(request: Request, files: list[UploadFile] = File(...)) -> JSONResponse:
    require_admin(request)
    saved = [await save_upload(file, SHARED_DIR) for file in files]
    return JSONResponse({"ok": True, "saved": saved})


@app.post("/api/upload/inbox")
async def upload_inbox(files: list[UploadFile] = File(...)) -> JSONResponse:
    saved = [await save_upload(file, INBOX_DIR) for file in files]
    return JSONResponse({"ok": True, "saved": saved})


@app.delete("/api/files/{space}/{filename}")
async def delete_file(space: Literal["shared", "inbox"], filename: str, request: Request) -> JSONResponse:
    require_admin(request)
    path = resolve_space(space) / safe_filename(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    path.unlink()
    return JSONResponse({"ok": True})


@app.get("/download/{space}/{filename}")
async def download_file(space: Literal["shared", "inbox"], filename: str, request: Request) -> FileResponse:
    if space == "inbox":
        require_admin(request)

    path = resolve_space(space) / safe_filename(filename)
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")

    return FileResponse(path, filename=path.name)


@app.get("/api/health")
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "ok": True,
            "shared_dir": str(SHARED_DIR),
            "inbox_dir": str(INBOX_DIR),
            "admin_required": bool(ADMIN_KEY),
            "max_upload_mb": MAX_UPLOAD_MB,
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=2345, reload=False)
