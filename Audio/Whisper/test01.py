from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.editor import VideoFileClip, CompositeAudioClip
import os
import pandas as pd
import numpy as np

if __name__ == '__main__':
    video_path = r"C:\Code\ML\Video\直播数据\whatnot直播视频\分段3_2025.11.25.08.24.20.mp4"
    audio_file_save_path = r"C:\Code\ML\Video\直播数据\whatnot直播视频\分段3.mp3"

    # 加载视频文件
    video = VideoFileClip(video_path)
    audio = video.audio

    # start_time = 1
    # end_time = 8
    # cut_audio = audio.subclip(start_time, end_time)

    audio.write_audiofile(audio_file_save_path)
    print('end')
