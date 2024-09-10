import torch
from transformers import WhisperForConditionalGeneration, AutoProcessor, pipeline
import os

from peft import LoraConfig, PeftModel, PeftConfig
import evaluate

from transformers import Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR





if __name__ == '__main__':
    model_id = "openai/whisper-large-v3"
    task = "transcribe"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    peft_model_id = r"D:\Code\ML\Project\untitled10\Audio\Whisper\reach-vb\test\checkpoint-100"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        # device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.to(device)
    # model.config.use_cache = True

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_path = r"D:\Code\ML\Audio\tomb.mp3"
    result = pipe(audio_path, return_timestamps=True)

    for data in result['chunks']:
        print("[%.2fs -> %.2fs] %s" % (data['timestamp'][0], data['timestamp'][1], data['text']))
