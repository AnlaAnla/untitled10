from transformers import WhisperForConditionalGeneration

model_name_or_path = "openai/whisper-large-v3"
task = "transcribe"

model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")