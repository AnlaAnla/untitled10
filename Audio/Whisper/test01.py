from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip ,CompositeAudioClip
import pypinyin
import pysrt
import json
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    video_path = r"D:\Code\ML\Video\1_8\t7.mp4"
    audio_file_save_path = r"D:\Code\ML\Video\1_8\1_8.mp3"

    # 加载视频文件
    video = VideoFileClip(video_path)
    audio = video.audio

    start_time = 1
    end_time = 8
    cut_audio = audio.subclip(start_time, end_time)


    cut_audio.write_audiofile(audio_file_save_path)
    print('end')
