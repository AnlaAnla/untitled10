import gradio as gr

def greet(name, instensity):
    return f"Hello, {name}! {instensity}"

demo = gr.Interface(
    fn=greet,
    inputs=['text', 'slider'],
    outputs=['text'],
)

demo.launch()