from datasets import load_dataset, Audio
import torch
import torchaudio
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer

model_name_or_path = "openai/whisper-large-v3"
task = "transcribe"
language = "Chinese"

feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)

class WhisperDataset(torch.utils.data.Dataset):
    def __init__(self, audio_paths, transcript_paths):
        self.audio_paths = audio_paths
        self.transcript_paths = transcript_paths

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        transcript_path = self.transcript_paths[idx]

        # 读取音频文件
        waveform, sample_rate = torchaudio.load(audio_path)

        # 读取文字标签
        with open(transcript_path, 'r') as f:
            transcript = f.read()

        # 对音频和文字标签进行预处理
        input_features = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt").input_features[0]
        transcript_tokens = tokenizer(transcript, return_tensors="pt").input_ids

        return input_features, transcript_tokens