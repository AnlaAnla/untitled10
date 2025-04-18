from datetime import datetime

from faster_whisper import WhisperModel
import srt
import moviepy.editor as mp
import datetime
from zhconv import convert
import pandas as pd
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
# model_size = "medium"
# model_size = r"D:\Code\ML\Model\Whisper\whisper-larev3turbp_2025Y_01M_03D_15h_03m_46s-ct2"

model = WhisperModel(model_size, device="cuda", compute_type="float16")


def transcribe_audio(audio_path):
    result = model.transcribe(audio_path,
                              # language='zh',
                              task="transcribe",

                              beam_size=5,
                              word_timestamps=True,
                              vad_filter=True,
                              # repetition_penalty=1.15,
                              # vad_parameters=vad_param,
                              # no_speech_threshold=0.2,
                              # max_initial_timestamp=9999999.0
                              # temperature=0,
                              # hotwords='Base'
                              )
    segments, info = result

    # 创建SRT字幕文件
    subtitles = []

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

    # for segment in segments:
    #     print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    #     text = segment.text.strip()
    #     text = convert(text, 'zh-cn')
    new_segments = []
    for segment in segments:
        words = segment.words
        current_segment_start = segment.start
        current_segment_text = ""
        for i in range(len(words)):
            current_segment_text += words[i].word
            # 设定一个阈值, 比如大于1秒就分割
            if i + 1 < len(words) and words[i + 1].start - words[i].end > 1.0:
                new_segments.append(
                    {
                        "start": current_segment_start,
                        "end": words[i].end,
                        "text": current_segment_text.strip()
                    }
                )
                current_segment_start = words[i + 1].start
                current_segment_text = ""
        new_segments.append(
            {
                "start": current_segment_start,
                "end": words[-1].end,
                "text": current_segment_text.strip()
            }
        )

    for segment in new_segments:
        print("[%.2fs -> %.2fs] %s" % (segment["start"], segment["end"], segment["text"]))


        # 写入srt
        start_time = datetime.timedelta(seconds=segment["start"])
        end_time = datetime.timedelta(seconds=segment["end"])
        subtitles.append(
            srt.Subtitle(
                index=len(subtitles) + 1,
                start=start_time,
                end=end_time,
                content=segment["text"],
            )
        )

    # 保存SRT字幕文件
    srt_file = srt.compose(subtitles)
    with open(srt_save_path, "w", encoding="utf-8") as f:
        f.write(srt_file)


if __name__ == '__main__':
    # audio_dir = r"D:\Code\ML\Audio\test_audio02"
    # for audio_name in os.listdir(audio_dir):
    #     audio_path = os.path.join(audio_dir, audio_name)
    #
    #     print(audio_name)
    #     transcribe_audio(audio_path)
    #     print("==" * 10)
    srt_save_path = r"D:\Code\ML\Project\untitled10\WebNetwork\Gradio\upload_downloda\temp\pokemon2\pokemon2.srt"

    transcribe_audio(r"D:\Code\ML\Project\untitled10\WebNetwork\Gradio\upload_downloda\temp\pokemon2\pokemon2.mp4")
