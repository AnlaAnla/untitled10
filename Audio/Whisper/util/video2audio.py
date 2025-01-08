import moviepy.editor as mp


def video2audio(video_path, audio_save_path):
    video = mp.VideoFileClip(video_path)
    # 提取音频文件
    audio = video.audio
    audio.write_audiofile(audio_save_path)
    print('Audio saved to {}'.format(audio_save_path))


if __name__ == '__main__':
    video_path = r"D:\Code\ML\Audio\0_2024Y_12M_30D_10h_56m_32s.mp4"
    audio_save_path = r"D:\Code\ML\Audio\0_2024Y_12M_30D_10h_56m_32s.mp3"
    video2audio(video_path, audio_save_path)
