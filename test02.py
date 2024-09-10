from moviepy.editor import VideoFileClip

video = VideoFileClip(r"D:\Code\ML\Video\video01.mp4")
audio = video.audio

audio.write_audiofile(r"D:\Code\ML\Audio\tomb.mp3")
print('end')