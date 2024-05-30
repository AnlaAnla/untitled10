from selenium import webdriver
import requests

# 启动浏览器
driver = webdriver.Chrome()

# 加载包含 blob URL 的网页
driver.get("blob:https://www.bilibili.com/video/BV1xx411x7qk/")

# 在浏览器中找到真实的视频文件 URL
video_url = driver.find_element_by_tag_name("video").get_attribute("src")

# 使用 requests 库下载视频文件
response = requests.get(video_url)
with open("video.mp4", "wb") as file:
    file.write(response.content)

# 关闭浏览器
driver.quit()