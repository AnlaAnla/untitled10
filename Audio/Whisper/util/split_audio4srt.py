from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip ,CompositeAudioClip
import pypinyin
import pysrt
import json
import os
import pandas as pd
import numpy as np

from torch.utils.hipify.hipify_python import meta_data


# 判断一段拼音是否在文本中
def is_have_pinyin(judge_text: str, judge_word:str):
    data = pypinyin.lazy_pinyin(judge_text)
    sentence = " ".join(data)
    if judge_word in sentence:
        return True
    return False


# 获取包含某个发音的字段，对应的时间和文本信息, 比如使用judge_word="mu bei"
def get_text_time(srt_path, judge_word=None):
    subs = pysrt.open(srt_path, encoding='utf-8')

    audio_text_datas = []
    for sub in subs:
        start_time = sub.start.to_time()  # 字幕开始时间
        end_time = sub.end.to_time()  # 字幕结束时间
        start_ms = (
                           start_time.hour * 3600 + start_time.minute * 60 + start_time.second) * 1000 + start_time.microsecond // 1000
        end_ms = (end_time.hour * 3600 + end_time.minute * 60 + end_time.second) * 1000 + end_time.microsecond // 1000
        start_ms /= 1000
        end_ms /= 1000

        text = sub.text  # 字幕文本
        result = {"start": start_ms, "end": end_ms, "text": text}
        if judge_word is not None:
            if is_have_pinyin(text, judge_word):
                # 对时间和文本进行处理
                audio_text_datas.append(result)
            else:
                continue
        else:
            audio_text_datas.append(result)

    return audio_text_datas



if __name__ == '__main__':
    video_path = r"D:\Code\ML\Video\card_video\2024_09_11 13_15_31.mp4"
    srt_path = r"D:\BaiduSyncdisk\card_video\2024_09_11 13_15_31.srt"

    data_save_path = r"D:\Code\ML\Audio\card_audio_data"

    # temp_path = "temp.mp3"

    # 加载视频文件
    video = VideoFileClip(video_path)
    audio = video.audio
    # audio.write_audiofile(temp_path)
    #
    # audio = AudioFileClip(temp_path)
    audio_text_datas = get_text_time(srt_path, judge_word=None)

    # 开始剪切的对应的字幕和音频
    meta_data = []
    for i, data in enumerate(audio_text_datas):
        text: str
        start_time, end_time, text = data['start'], data['end'], data['text']

        print(data)
        # if "木杯" in text:
        #     text = text.replace('木杯', '墓碑')
        if len(text) < 2:
            continue

        cut_audio = audio.subclip(start_time, end_time)

        audio_save_name = f"data/audio{i}.mp3"
        info_save_name = f"audio{i}.json"
        # info = {
        #     "audio":{
        #         "path": audio_save_name
        #     },
        #     "sentence": str(text),
            # "language": "Chinese",
            # "duration": str(cut_audio.duration)
        # }
        # info = json.dumps(info , ensure_ascii=False)
        # print(info)

        # 写入新的音频文件
        audio_file_save_path = os.path.join(data_save_path, audio_save_name)
        # info_file_save_path = os.path.join(data_save_path, info_save_name)

        cut_audio.write_audiofile(audio_file_save_path)
        meta_data.append([audio_save_name, str(text)])
        # with open(info_file_save_path, "w", encoding="utf-8") as f:
        #     f.write(info)

        print(f"{i}: [{start_time}, {end_time}] : {text}")

    print(meta_data)
    meta_data = np.array(meta_data)
    meta_data = pd.DataFrame(meta_data, columns=['file_name', 'sentence'])
    print(meta_data)
    meta_data.to_csv(os.path.join(data_save_path, "pre_metadata.csv"), encoding='utf-8', index=False)

    print('end')

