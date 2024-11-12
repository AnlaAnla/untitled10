from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts",
                       device="cuda:0")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
# You can replace this embedding with your own as well.

speech = synthesiser("你好, 今天天气不错呀", forward_params={"speaker_embeddings": speaker_embedding})

sf.write("speech2.wav", speech["audio"], samplerate=speech["sampling_rate"])
print('end')
print("*"*10)