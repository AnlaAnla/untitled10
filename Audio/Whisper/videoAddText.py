from datetime import datetime

from faster_whisper import WhisperModel
import srt
import moviepy.editor as mp
import datetime
from zhconv import convert
import os
import re

import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
# model_size = "medium"
model_size = r"D:\Code\ML\Model\Whisper\whisper-larev3turbp_2025Y_01M_03D_15h_03m_46s-ct2"

model = WhisperModel(model_size, device="cuda", compute_type="float16")

audio_path = r"D:\Code\ML\Audio\test_audio01\tt601.mp3"
# 加载视频文件
# video = mp.VideoFileClip(r"D:\Code\ML\Video\card_video\2024_09_11 13_15_31.mp4")

# 提取音频文件
# audio = video.audio


# 保存音频文件为WAV格式
# audio.write_audiofile(audio_path)
# print("保存音频文件")



def split_segment_by_punctuation(segment):
    """
    根据空格和标点符号将 segment 分割成更小的片段。

    Args:
        segment: fast-whisper 生成的 segment 对象。

    Returns:
        一个列表，包含分割后的片段，每个片段是一个字典，包含 start、end 和 text 键。
    """
    words = segment.words
    if not words:
        return []

    split_segments = []
    current_segment = {"start": words[0].start, "end": words[0].end, "text": words[0].word}

    for i in range(1, len(words)):
        word = words[i]
        prev_word = words[i - 1]

        # 判断是否需要分割
        if re.search(r"[，。？！、\s]$", prev_word.word) or (word.start - prev_word.end >= 0.5): #根据情况调整这里的分割阈值
            # 分割并保存当前片段
            current_segment["end"] = prev_word.end
            split_segments.append(current_segment)
            # 开始新的片段
            current_segment = {"start": word.start, "end": word.end, "text": word.word}
        else:
            # 将单词添加到当前片段
            current_segment["text"] += word.word
            current_segment["end"] = word.end

    # 添加最后一个片段
    split_segments.append(current_segment)

    return split_segments


vad_param = {
    "threshold": 0.5,
    "min_speech_duration_ms": 2500,
    "min_silence_duration_ms": 100,
    "max_speech_duration_s": 30,
    "speech_pad_ms": 400
}

result = model.transcribe(audio_path,
                          language='zh',
                          task="transcribe",
                          beam_size=5,
                          word_timestamps=True,
                          # vad_filter=True,
                          repetition_penalty=1.2,
                          # vad_parameters=vad_param,
                          no_speech_threshold=0.4,
                          # max_initial_timestamp=9999999.0
                          # temperature=0,
                          )
segments, info = result

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

# 创建SRT字幕文件
subtitles = []
# start_time = datetime.timedelta()

t1 = time.time()

# for segment in segments:
#     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
#     text = segment.text.strip()
#     text = convert(text, 'zh-cn')
#     if text:
#         start_time = datetime.timedelta(seconds=segment.start)
#         end_time = datetime.timedelta(seconds=segment.end)
#         subtitles.append(
#             srt.Subtitle(
#                 index=len(subtitles) + 1,
#                 start=start_time,
#                 end=end_time,
#                 content=text,
#             )
#         )


# 显示方法2
for segment in segments:
    split_segments = split_segment_by_punctuation(segment)
    for split_segment in split_segments:
        print("[%.2fs -> %.2fs] %s" % (split_segment["start"], split_segment["end"], split_segment["text"]))

t2 = time.time()

print('cost time: %.3fs' % (t2 - t1))
# 保存SRT字幕文件
# srt_file = srt.compose(subtitles)
# with open(r"D:\BaiduSyncdisk\t7.srt", "w", encoding="utf-8") as f:
#     f.write(srt_file)

print("保存srt字幕")
