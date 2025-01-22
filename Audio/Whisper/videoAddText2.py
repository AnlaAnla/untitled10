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

# model_size = "deepdml/faster-whisper-large-v3-turbo-ct2"
# model_size = "medium"
model_size = r"D:\Code\ML\Model\Whisper\whisper-larev3turbp_2025Y_01M_03D_15h_03m_46s-ct2"

model = WhisperModel(model_size, device="cuda", compute_type="float16")


def transcribe_audio(audio_path):
    result = model.transcribe(audio_path,
                              language='zh',
                              task="transcribe",

                              beam_size=5,
                              word_timestamps=True,
                              vad_filter=True,
                              repetition_penalty=1.2,
                              # vad_parameters=vad_param,
                              # no_speech_threshold=0.2,
                              # max_initial_timestamp=9999999.0
                              # temperature=0,
                              hotwords='Base'
                              )
    segments, info = result

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


if __name__ == '__main__':
    # audio_dir = r"D:\Code\ML\Audio\test_audio02"
    # for audio_name in os.listdir(audio_dir):
    #     audio_path = os.path.join(audio_dir, audio_name)
    #
    #     print(audio_name)
    #     transcribe_audio(audio_path)
    #     print("==" * 10)

    transcribe_audio(r"D:\Code\ML\Audio\test_audio02\2_tt657_2025Y_01M_14D_18h_13m_16s.mp4.mp3")
