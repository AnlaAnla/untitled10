from datetime import datetime

import whisper
from faster_whisper import WhisperModel
import srt
import moviepy.editor as mp
import datetime
from zhconv import convert
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
# model_size = "medium"

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
vad_param = {
    "threshold": 0.5,
    "min_speech_duration_ms": 1000,
    "min_silence_duration_ms": 100,
    "max_speech_duration_s": 30,
    "speech_pad_ms": 2000
}

result = model.transcribe(audio_path, beam_size=5, word_timestamps=True,
                          vad_filter=True,
                          vad_parameters=vad_param,
                          no_speech_threshold=0.2,
                          max_initial_timestamp=9999999.0)
segments, info = result

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# 创建SRT字幕文件
subtitles = []
# start_time = datetime.timedelta()

t1 = time.time()

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

t2 = time.time()

print('cost time: %.3fs' % (t2 - t1))
# 保存SRT字幕文件
# srt_file = srt.compose(subtitles)
# with open(r"D:\BaiduSyncdisk\t7.srt", "w", encoding="utf-8") as f:
#     f.write(srt_file)

print("保存srt字幕")
