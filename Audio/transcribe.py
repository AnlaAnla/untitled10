from faster_whisper import WhisperModel
import json
import os
import time

# 配置
audio_files = ["temp/分段1.mp3", "temp/分段2.mp3", "temp/分段3.mp3"]
output_dir = "./transcripts"
model_size = "large-v3"  # 或者 "medium.en" 以获得更快的速度

# 1. 加载模型 (compute_type="float16" 利用显卡加速)
print(f"正在加载 Faster-Whisper 模型: {model_size}...")
# 如果你的显卡显存较小(比如小于8G)，可以将 compute_type 改为 "int8_float16" 或 "int8"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for index, file_path in enumerate(audio_files):
    start_time = time.time()
    print(f"[{index + 1}/{len(audio_files)}] 正在识别: {file_path} ...")

    # 2. 执行识别
    # beam_size=5 是默认值，精度较高
    segments, info = model.transcribe(file_path, beam_size=5, language="en")

    # 3. 提取结果
    # faster-whisper 返回的是一个生成器(generator)，所以必须遍历它才会开始真正的计算
    results = []
    for segment in segments:
        results.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })
        # 可以在这里打印进度，比如 print(f"{segment.start}s: {segment.text}")

    # 4. 保存
    output_filename = os.path.join(output_dir, f"transcript_{index + 1}.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump({"filename": file_path, "segments": results}, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"完成！耗时: {elapsed:.2f}秒. 结果保存至: {output_filename}")

print("所有任务处理完毕。")