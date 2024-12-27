import requests

# 服务器 IP 地址
server_ip = "127.0.0.1"
# 服务器端口号
server_port = 8888

# 获取可下载的文件列表
# res = requests.get(f"http://{server_ip}:{server_port}/files")
# files = res.json()["files"]
# print(f"Available files: {files}")

# 下载文件
# file_name = "config.json"
# res = requests.get(f"http://{server_ip}:{server_port}/files/{file_name}")
# if "error" in res.json():
#     print(res.json()["error"])
# else:
#     file_path = res.json()["file_path"]
#     with open(file_name, "wb") as f:
#         f.write(res.content)
#     print(f"File downloaded: {file_path}")

# ================================ 下载main的

res = requests.get(f"http://{server_ip}:{server_port}/download")
if "error" in res.json():
    print(res.json()["error"])
else:
    file_path = res.json()["file_path"]
    with open("sentence_judge_bert03.zip", "wb") as f:
        f.write(res.content)
    print(f"File downloaded: {file_path}")
