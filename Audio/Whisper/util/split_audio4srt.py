from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip ,CompositeAudioClip
import pypinyin
import pysrt
import json
import os


# 判断一段拼音是否在文本中
def is_have_pinyin(judge_text: str, judge_word:str):
    data = pypinyin.lazy_pinyin(judge_text)
    sentence = " ".join(data)
    if judge_word in sentence:
        return True
    return False


# 获取包含某个发音的字段，对应的时间和文本信息
def get_text_time(srt_path, judge_word='mu bei'):
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
        if is_have_pinyin(text, judge_word):
            # 对时间和文本进行处理
            result = {"start": start_ms, "end": end_ms, "text": text}
            audio_text_datas.append(result)

    return audio_text_datas



if __name__ == '__main__':
    video_path = r"D:\Code\ML\Video\video01.mp4"
    srt_path = r"D:\Code\ML\Video\output_subtitles.srt"

    audio_data_save_path = r"D:\Code\ML\Audio\Data1\dataset"
    json_data_save_path = r"D:\Code\ML\Audio\Data1"

    # temp_path = "temp.mp3"

    # 加载视频文件
    video = VideoFileClip(video_path)
    audio = video.audio
    # audio.write_audiofile(temp_path)
    #
    # audio = AudioFileClip(temp_path)
    audio_text_datas = get_text_time(srt_path, judge_word='mu bei')

    # 开始剪切的对应的字幕和音频
    for i, data in enumerate(audio_text_datas):
        text: str
        start_time, end_time, text = data['start'], data['end'], data['text']

        print(data)
        if "木杯" in text:
            text = text.replace('木杯', '墓碑')

        cut_audio = audio.subclip(start_time, end_time)

        audio_save_name = f"audio{i}.mp3"
        info_save_name = f"audio{i}.json"
        info = {
            "audio":{
                "path": audio_save_name
            },
            "sentence": str(text),
            "language": "Chinese",
            "duration": str(cut_audio.duration)
        }
        info = json.dumps(info , ensure_ascii=False)
        # print(info)

        # 写入新的音频文件
        audio_file_save_path = os.path.join(audio_data_save_path, audio_save_name)
        info_file_save_path = os.path.join(json_data_save_path, info_save_name)

        cut_audio.write_audiofile(audio_file_save_path)
        with open(info_file_save_path, "w", encoding="utf-8") as f:
            f.write(info)

        print(f"{i}: [{start_time}, {end_time}] : {text}")

    print('end')

