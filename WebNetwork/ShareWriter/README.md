# ShareWriter — 简易同步笔记本

这是一个基于 FastAPI 的轻量同步笔记本示例（无数据库）。

运行:

```bash
pip install -r requirements.txt
python mian_share_writer.py
# 访问 http://localhost:8000
```

说明:

- 使用 WebSocket 在多个客户端之间广播文本内容。
- 文档保存在内存中，服务器重启后会重置。
