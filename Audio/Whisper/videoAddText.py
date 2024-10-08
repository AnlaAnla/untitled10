from datetime import datetime

import whisper
from faster_whisper import WhisperModel
import srt
import moviepy.editor as mp
import datetime
from zhconv import convert
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# model_size = "D:\Code\ML\Model\huggingface\whisper-largev3-ct2"
model_size = "medium"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

audio_path = r"D:\Code\ML\Audio\tomb.mp3"
# 加载视频文件
# video = mp.VideoFileClip(r"D:\Code\ML\Video\card_video\2024_09_11 13_15_31.mp4")

# 提取音频文件
# audio = video.audio


# 保存音频文件为WAV格式
# audio.write_audiofile(audio_path)
# print("保存音频文件")

result = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
segments, info = result


print("Detected language '%s' with probability %f" % (info.language, info.language_probability))


# 创建SRT字幕文件
subtitles = []
# start_time = datetime.timedelta()

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    text = segment.text.strip()
    text = convert(text, 'zh-cn')
    if text:
        start_time = datetime.timedelta(seconds=segment.start)
        end_time = datetime.timedelta(seconds=segment.end)
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
with open(r"D:\Code\ML\Video\card_video\2024_09_11 13_15_31.srt", "w", encoding="utf-8") as f:
    f.write(srt_file)

print("保存srt字幕")
