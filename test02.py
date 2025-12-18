import json

def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02.0f}"

# 读取文件
with open(r"C:\Code\ML\Project\untitled10\Audio\temp\transcripts\backyardhits1.json", 'r', encoding='utf-8') as f:
    data = json.load(f)

# 转换
for segment in data['segments']:
    segment['start'] = format_time(segment['start'])
    segment['end'] = format_time(segment['end'])

    print(segment)

# 保存到新文件
with open(r"C:\Code\ML\Project\untitled10\Audio\temp\transcripts\backyardhits1.txt", 'w', encoding='utf-8') as f:
    for segment in data['segments']:
        break_text = "\n"
        text = f"{segment['start']}{break_text}{segment['text']}{break_text}"
        f.write(text)
        f.write(break_text)

# with open(r"C:\Code\ML\Project\untitled10\Audio\temp\transcripts\_backyardhits1.json", 'w', encoding='utf-8') as f:
#     json.dump(data, f, indent=2, ensure_ascii=False)
#
# print("转换完成，已保存为 output_formatted.json")