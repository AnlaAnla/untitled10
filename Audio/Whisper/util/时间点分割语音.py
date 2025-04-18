from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip ,CompositeAudioClip
import pypinyin
import pysrt
import json
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    video_path = r"D:\Code\ML\Project\untitled10\WebNetwork\Gradio\upload_downloda\temp\pokemon\音色0-30.mp3"
    audio_file_save_path = r"D:\Code\ML\Project\untitled10\WebNetwork\Gradio\upload_downloda\temp\pokemon\大家好,这里是卡片驿站的宝可梦专区 今天的视频将会介绍到之前发布的产品中挑选两个盒子.mp3"

    # 加载视频文件
    # video = VideoFileClip(video_path)
    # audio = video.audio
    audio = AudioFileClip(video_path)

    start_time = 0
    end_time = 7.38
    cut_audio = audio.subclip(start_time, end_time)


    cut_audio.write_audiofile(audio_file_save_path)
    print('end')
