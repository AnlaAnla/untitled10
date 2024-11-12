import os
import subprocess


def download_m3u8_and_convert_to_mp4(m3u8_url, output_path):
    # 生成一个临时的ts文件路径
    temp_file = "temp/output.mp4"

    # 调用ffmpeg来下载m3u8并转换为mp4
    command = [
        "ffmpeg",
        "-i", m3u8_url,  # 输入m3u8文件的URL
        "-c", "copy",  # 直接拷贝，不重新编码
        temp_file  # 输出mp4文件
    ]

    # 执行ffmpeg命令
    subprocess.run(command, check=True)
    print("sleep the time is so long!!!")

    # 将文件保存到指定路径
    os.rename(temp_file, output_path)
    print(f"文件已保存到: {output_path}")


# 使用函数下载并转换文件
m3u8_url = "https://upyun.luckly-mjw.cn/Assets/media-source/example/media/index.m3u8"
output_path = "temp/output_video.mp4"  # 指定输出文件的路径

download_m3u8_and_convert_to_mp4(m3u8_url, output_path)
