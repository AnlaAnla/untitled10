import os
from faster_whisper import WhisperModel


def format_timestamp(seconds: float):
    """
    将秒数转换为 SRT 时间戳格式 (HH:MM:SS,mmm)
    """
    whole_seconds = int(seconds)
    # milliseconds = int((seconds - whole_seconds) * 1000)

    hours = whole_seconds // 3600
    minutes = (whole_seconds % 3600) // 60
    secs = whole_seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def transcribe_mp3_to_srt(mp3_path, model_size="large-v3", device="cuda", compute_type="int8_float16",
                          language=None):
    print(f"正在加载模型: {model_size} ({compute_type})...")

    # 1. 初始化模型
    # 这里是核心：使用 cuda 和 int8_float16 达到速度与精度的平衡
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    print(f"正在转录音频: {mp3_path} ...")

    # 2. 开始转录
    # beam_size=5 是官方推荐的精度设置
    # vad_filter=True 会自动过滤静音片段，极大提升长音频的处理速度
    segments, info = model.transcribe(
        mp3_path,
        beam_size=5,
        vad_filter=True,
        language=language
    )

    print(f"检测到语言: {info.language}, 置信度: {info.language_probability:.2f}")

    # 3. 输出文件名
    srt_filename = os.path.splitext(mp3_path)[0] + ".txt"

    # 4. 写入 SRT 文件
    # 注意：segments 是一个生成器，只有在遍历时才会真正开始计算（流式处理）
    with open(srt_filename, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start_time = format_timestamp(segment.start)
            end_time = format_timestamp(segment.end)
            text = segment.text.strip()

            # 写入 SRT 格式
            # f.write(f"{i}\n")
            # f.write(f"{start_time} --> {end_time}\n")
            # f.write(f"{text}\n\n")

            # txt 格式
            f.write(f"{start_time}\n")
            f.write(f"{text}\n\n")

            # 可选：实时打印进度
            print(f"[{start_time} -> {end_time}] {text}")

    print(f"\n✅ 提取完成！字幕已保存为: {srt_filename}")


if __name__ == "__main__":
    # 替换为你的 mp3 文件路径
    audio_file = "/home/martin/ML/RemoteProject/untitled10/Audio/temp/audio/2026_02_25 16_47_46.mp3"

    if os.path.exists(audio_file):
        transcribe_mp3_to_srt(audio_file, compute_type="default")
    else:
        print(f"找不到文件: {audio_file}")
        print('==')
