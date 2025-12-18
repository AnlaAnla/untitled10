import whisper
import json
import os
import time
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ================= 配置区域 =================
# 输入 MP3 目录
input_dir = "./temp/audio"
# 输出 JSON 目录
output_dir = "./temp/transcripts"

# 模型大小: "large-v3", "medium", "small", "base"
# 注意: 官方 large-v3 需要约 10GB 显存。如果显存报错，请改用 "medium"
model_size = "large-v3"
# ===========================================

# 1. 检查并创建输出目录
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"已创建输出目录: {output_dir}")

# 2. 获取目录下所有 mp3 文件
mp3_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mp3')]

if not mp3_files:
    print(f"警告: 在 {input_dir} 中没有找到 .mp3 文件。")
    exit()

# 3. 加载模型
print(f"正在加载 OpenAI-Whisper 模型: {model_size}...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"运行设备: {device}")

try:
    # 官方库会自动下载模型到 ~/.cache/whisper
    model = whisper.load_model(model_size, device=device)
except Exception as e:
    print(f"加载模型出错: {e}")
    print("建议检查显存或尝试更小的模型 (如 'medium')")
    exit()

print(f"准备开始，共发现 {len(mp3_files)} 个音频文件。")

# 4. 批量循环处理
for index, filename in enumerate(mp3_files):
    file_path = os.path.join(input_dir, filename)
    file_name_without_ext = os.path.splitext(filename)[0]
    output_filename = os.path.join(output_dir, f"{file_name_without_ext}.json")

    start_time = time.time()
    print(f"[{index + 1}/{len(mp3_files)}] 正在识别: {filename} ...")

    # 5. 执行识别
    # language=None 表示自动检测。
    # 如果你是纯中文，建议改为 language="zh"；纯英文改为 language="en"
    try:
        result = model.transcribe(
            file_path,
            beam_size=5,
            language=None, # None = 自动检测
            fp16=True if device == "cuda" else False, # 显卡加速时开启半精度
            verbose=False # 设为 True 可以在控制台实时看到吐字
        )
    except Exception as e:
        print(f"   -> 识别文件出错: {e}")
        continue

    # 获取检测到的语言 (官方库结果字典里包含了 language 字段)
    detected_lang = result.get('language', 'unknown')
    print(f"   -> 检测到语言: {detected_lang}")

    # 6. 提取结果
    # 官方库的 segments 结构和 faster-whisper 略有不同，是一个字典列表
    formatted_segments = []
    for segment in result['segments']:
        formatted_segments.append({
            "start": round(segment['start'], 2),
            "end": round(segment['end'], 2),
            "text": segment['text'].strip()
        })

    # 7. 保存结果
    with open(output_filename, "w", encoding="utf-8") as f:
        data = {
            "filename": filename,
            "language": detected_lang,
            "segments": formatted_segments,
            # "full_text": result['text'].strip()
        }
        json.dump(data, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    print(f"   -> 完成！耗时: {elapsed:.2f}秒. 保存至: {output_filename}")

print("-" * 30)
print(f"所有任务处理完毕。输出目录: {output_dir}")