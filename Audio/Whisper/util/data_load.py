from datasets import load_dataset, DatasetDict, Audio
import os
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

model_name_or_path = "openai/whisper-large-v3"
task = "transcribe"

dataset_name = "mozilla-foundation/common_voice_13_0"
language = "Hindi"
language_abbr = "hi" # Short hand code for the language we want to fine-tune

common_voice = DatasetDict()

# common_voice["train"] = load_dataset(dataset_name, language_abbr, split="train+validation", use_auth_token=True)
common_voice["test"] = load_dataset(dataset_name, language_abbr, split="test", use_auth_token=True, trust_remote_code=True)

print(common_voice)

common_voice = common_voice.remove_columns(
    ["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes", "variant"]
)

print(common_voice)



feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language=language, task=task)
processor = WhisperProcessor.from_pretrained(model_name_or_path, language=language, task=task)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch


common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["test"], num_proc=1)

print(common_voice)
print()