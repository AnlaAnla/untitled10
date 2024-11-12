import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import datetime
import srt
from zhconv import convert

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# model_id = "openai/whisper-large-v3"
model_id = "openai/whisper-medium"

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,

    # max_new_tokens=128,
    # chunk_length_s=30,
    # batch_size=8,
    # torch_dtype=torch_dtype,
    device=device,
)

generate_kwargs = {"task": "transcribe", "num_beams": 1}
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]

audio_path = r"D:\BaiduSyncdisk\t7.mp3"
# result = pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)
result = pipe(audio_path, return_timestamps=True, generate_kwargs=generate_kwargs)

# print()
# print(result)
# print()
# print(result["text"])

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
# srt_file = srt.compose(subtitles)
# with open(r"D:\Code\ML\Audio\card_audio_data\2024_09_11 13_15_31.srt", "w", encoding="utf-8") as f:
#     f.write(srt_file)

print("保存srt字幕")
