import gradio as gr
from transformers import pipeline
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-medium"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,

    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=4,
    # torch_dtype=torch_dtype,
    device=device,
)
generate_kwargs = {"task": "transcribe", "num_beams": 1}


def transcribe(filepath):

    return pipe(filepath, generate_kwargs=generate_kwargs)["text"]


demo = gr.Interface(
    transcribe,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=gr.Textbox(),
)

demo.launch(server_name='0.0.0.0', debug=True)
