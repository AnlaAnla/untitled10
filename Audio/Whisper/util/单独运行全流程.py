import datasets
import torchaudio
import torch
import os
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperForConditionalGeneration


model_name_or_path = "openai/whisper-large-v3"
task = "transcribe"
language = "Chinese"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, device_map=device)
model.eval()


for i in range(15):
    name = f"audio{i}.mp3"
    waveform, sample_rate = torchaudio.load(os.path.join(r"D:\Code\ML\Audio\Data1\dataset", name))
    new_sample_rate = 16000
    resampled_waveform = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(waveform)

    # 将双声道音频转换为单声道
    resampled_waveform = resampled_waveform.mean(dim=0, keepdim=True)[0]

    input_features = feature_extractor(resampled_waveform, sampling_rate=new_sample_rate, return_tensors="pt").input_features[0]

    # encode target text to label ids
    # label = tokenizer("今天天气不错，天气很好", return_tensors="pt").input_ids


    with torch.no_grad():
        # 增加一个维度来表示通道数
        input_features = input_features.unsqueeze(0)
        input_features = input_features.to(device)
        print("shape: ",input_features.shape)
        transcription = model.generate(input_features)

    transcription_text = tokenizer.decode(transcription[0], skip_special_tokens=True)

    print(transcription_text)
    # print(input_features)
    # print(label)
    print()