from datetime import datetime
from faster_whisper import WhisperModel
from zhconv import convert
import os
import pandas as pd
import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def process_pd_data(data_path):
    # 首先创建一个布尔掩码,用于识别需要替换为 0 的值
    data = pd.read_csv(data_path)
    mask_zero = data['label'].isna() | (data['label'].str.strip() == '')
    data = data.drop(data[mask_zero].index)
    data.reset_index(drop=True, inplace=True)  # 重置索引

    return data


def test(model_size, data_path):
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    data = process_pd_data(data_path)

    vad_param = {
        "threshold": 0.5,
        "min_speech_duration_ms": 250,
        "min_silence_duration_ms": 100,
        "max_speech_duration_s": 30,
        "speech_pad_ms": 2000
    }

    # result = model.transcribe(audio_path, beam_size=5, word_timestamps=False,
    #                           vad_filter=True,
    #                           vad_parameters=vad_param,
    #                           no_speech_threshold=0.4,
    #                           max_initial_timestamp=9999999.0)
    # segments, info = result
    # for segment in segments:
    #     text = segment.text.strip()
    #     text = convert(text, 'zh-cn')
    #     print(text)


if __name__ == '__main__':
    model_size = r"D:\Code\ML\Model\Whisper\checkpoint-100-2024Y_11M_29D_11h_55m_31s"
    data_path = r"D:\Code\ML\Text\Classify\judge_data\judge_metadata.csv"

    test(model_size, data_path)
