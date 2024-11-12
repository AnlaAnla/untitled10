import gradio as gr
import time



# 定义音频处理函数
def transcribe_audio():
    transcript = ""

    for i in range(50):
        transcript += f"{i}--\n"
        time.sleep(0.2)
        yield transcript


def get_data():
    yield from transcribe_audio()

# 创建 Gradio 界面
with gr.Blocks() as demo:
    btn = gr.Button("Click")
    text_output = gr.Textbox()  # 显示识别的文本

    # 调用 transcribe_audio，动态更新 text_output
    btn.click(get_data, outputs=text_output)

# 启动 Gradio 界面
demo.launch()
