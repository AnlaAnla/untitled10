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
                              )
    segments, info = result

    print("Detected language '%s' with probability %f" % (info.language, info.language_probability))


    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        # text = segment.text.strip()
        # text = convert(text, 'zh-cn')



if __name__ == '__main__':
    audio_dir = r"D:\Code\ML\Audio\card_audio_data02\audio"
    for audio_name in os.listdir(audio_dir):
        audio_path = os.path.join(audio_dir, audio_name)

        print(audio_name)
        transcribe_audio(audio_path)
        print("=="*10)