from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import asyncio
import json
import uuid

app = FastAPI()
# Use paths relative to this file so templates/static are found when started from any cwd
BASE_DIR = Path(__file__).parent.resolve()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# In-memory shared document state (no database)
_doc = {"content": "", "version": 0}
_clients = set()
_lock = asyncio.Lock()


@app.get("/")
async def index(request: Request):
	return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
	await websocket.accept()
	client_id = str(uuid.uuid4())
	async with _lock:
		_clients.add(websocket)
	try:
		# Send initial document state
		await websocket.send_text(json.dumps({
			"type": "init",
			"content": _doc["content"],
			"version": _doc["version"],
			"client_id": client_id,
		}))

		while True:
			data = await websocket.receive_text()
			msg = json.loads(data)
			if msg.get("type") == "update":
				content = msg.get("content", "")
				async with _lock:
					_doc["content"] = content
					_doc["version"] += 1
					version = _doc["version"]
					# Broadcast to other connected clients
					to_remove = []
					for ws in list(_clients):
						if ws is websocket:
							continue
						try:
							await ws.send_text(json.dumps({
								"type": "update",
								"content": content,
								"version": version,
								"from": client_id,
							}))
						except Exception:
							to_remove.append(ws)
					for ws in to_remove:
						_clients.discard(ws)

	except WebSocketDisconnect:
		async with _lock:
			_clients.discard(websocket)
	except Exception:
		async with _lock:
			_clients.discard(websocket)


if __name__ == "__main__":
	import uvicorn

	uvicorn.run("mian_share_writer:app", host="0.0.0.0", port=8000)

