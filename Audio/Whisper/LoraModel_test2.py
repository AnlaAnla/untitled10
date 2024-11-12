import torch
from transformers import WhisperForConditionalGeneration, AutoProcessor, pipeline
import os

from peft import LoraConfig, PeftModel, PeftConfig
import evaluate
import datetime
import srt
from zhconv import convert

from transformers import Seq2SeqTrainingArguments, BitsAndBytesConfig
from transformers import Seq2SeqTrainer, TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

if __name__ == '__main__':
    model_id = "openai/whisper-medium"
    task = "transcribe"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    peft_model_id = r"D:\Code\ML\Model\Lora\checkpoint-100-2024Y_10M_08D_17h_43m_12s\adapter_model"
    peft_config = PeftConfig.from_pretrained(peft_model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id,
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        # device_map="cuda:0"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)
    model.to(device)
    model.config.use_cache = True

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    audio_path = r"D:\Code\ML\Audio\t7.mp3"
    result = pipe(audio_path, return_timestamps=True)

    # for data in result['chunks']:
    #     print("[%.2fs -> %.2fs] %s" % (data['timestamp'][0], data['timestamp'][1], data['text']))

    # 创建SRT字幕文件
    subtitles = []
    # start_time = datetime.timedelta()
    temp_end_time = 0
    chunks_start_time = 0
    for chunk in result['chunks']:
        # 每30s为一个周期, 进行正确的时间累加
        if temp_end_time > chunk['timestamp'][0]:
            chunks_start_time += 30
            print('')
        temp_end_time = chunk['timestamp'][1]

        start_time = chunk['timestamp'][0] + chunks_start_time
        end_time = chunk['timestamp'][1] + chunks_start_time
        text = convert(chunk['text'], 'zh-cn')

        print("[%.2fs -> %.2fs] %s" % (start_time, end_time, text))

        if text:
            start_time = datetime.timedelta(seconds=start_time)
            end_time = datetime.timedelta(seconds=end_time)
            subtitles.append(
                srt.Subtitle(
                    index=len(subtitles) + 1,
                    start=start_time,
                    end=end_time,
                    content=text,
                )
            )

    # 保存SRT字幕文件
    srt_file = srt.compose(subtitles)
    with open(r"D:\Code\ML\Video\merged_video.srt", "w", encoding="utf-8") as f:
        f.write(srt_file)

    print("保存srt字幕")
