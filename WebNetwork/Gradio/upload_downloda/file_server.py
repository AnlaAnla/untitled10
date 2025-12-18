import gradio as gr
import os

# 指定临时文件夹路径
gradio_temp_folder = "temp"
os.makedirs(gradio_temp_folder, exist_ok=True)

# 服务器文件路径（写死）
server_file_path = "temp/1.txt"
# server_file_path = "temp/pokemon2.zip"
# 如果文件不存在，创建一个示例文件
if not os.path.exists(server_file_path):
    with open(server_file_path, "w") as f:
        f.write("This is a sample file on the server.")  

# 服务器上传文件保存路径
upload_dir = "temp"  # 替换成你的上传目录
# 如果目录不存在，创建目录
os.makedirs(upload_dir, exist_ok=True)


def download_file():
    """下载服务器文件"""
    return server_file_path


def upload_file(file):
    """上传文件到服务器"""
    if file is None:
        return "没有放入文件"

    file_path = file.name

    filename = os.path.basename(file_path)
    destination_path = os.path.join(upload_dir, filename)

    # 避免文件名冲突，如果文件已存在，添加编号
    base, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(destination_path):
        destination_path = os.path.join(upload_dir, f"{base}_{counter}{ext}")
        counter += 1

    with open(file_path, "rb") as f_in, open(destination_path, "wb") as f_out:
        f_out.write(f_in.read())

    return f"文件 '{os.path.basename(destination_path)}' 上传成功!"


# Gradio 界面
with gr.Blocks(title="File Downloader & Uploader", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 文件上传下载页面
        来吧, 上传下载把!
        """
    )
    with gr.Row():
        with gr.Column():
            gr.Label("下载服务器文件", show_label=True)
            server_filename_label = gr.Label(
                value=os.path.basename(server_file_path), label="Server File:"
            )
            download_button = gr.Button("Download", variant="primary")
        with gr.Column():
            gr.Label("上传文件到服务器", show_label=True)
            upload_file_input = gr.File(label="上传文件", interactive=True)
            upload_button = gr.Button("上传", variant="primary")
            upload_status_label = gr.Label(label="上传状态")

    # 连接事件
    download_button.click(download_file, outputs=gr.File(label="下载文件"))
    upload_button.click(upload_file, inputs=upload_file_input, outputs=upload_status_label)

demo.launch(server_name='0.0.0.0', server_port=2345)
