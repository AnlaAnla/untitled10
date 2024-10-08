import gradio as gr


def train_yolov8(dataset_path, model_path, training_params):
    # 这里是您的训练代码
    # 您可以使用dataset_path、model_path和training_params作为输入
    # 并在此处编写训练循环和日志记录

    # 为了演示,我们将模拟训练过程
    training_log = "Training started...\n\n"
    for epoch in range(1, 101):
        loss = 2.34 - (epoch * 0.02)
        training_log += f"Epoch {epoch}/100: Loss = {loss:.2f}\n"

    training_log += "Training completed."
    return training_log


with gr.Blocks() as demo:
    gr.Markdown("# YOLOv8 Training Interface")

    with gr.Row():
        with gr.Column():
            dataset_path = gr.Textbox(label="Dataset Path")
            model_path = gr.Textbox(label="Model Path")
            training_params = gr.Textbox(label="Training Parameters", lines=4)
            train_button = gr.Button("Start Training")

        with gr.Column():
            output = gr.Textbox(label="Training Output", lines=10)

    train_button.click(fn=train_yolov8, inputs=[dataset_path, model_path, training_params], outputs=output)

demo.launch()
