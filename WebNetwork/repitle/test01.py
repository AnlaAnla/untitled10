from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# 设置 WebDriver，确保 ChromeDriver 路径正确
service = Service()  # 替换为你的 chromedriver 路径
driver = webdriver.Chrome(service=service)

# 访问百度首页
driver.get("https://www.baidu.com")

# 搜索框输入 "中国百度百科"
search_box = driver.find_element(By.ID, "kw")  # 找到搜索框
search_box.send_keys("中国百度百科")
search_box.send_keys(Keys.RETURN)  # 模拟按下回车键

# 等待页面加载
time.sleep(2)

# 点击搜索结果中的第一个链接（百度百科链接）
first_result = driver.find_element(By.XPATH, '''//*[@id="1"]/div/h3/a/div/div/p/span/span''')
first_result.click()

# 等待页面加载
time.sleep(2)

# 滚动页面
driver.execute_script("window.scrollTo(10, document.body.scrollHeight);")
time.sleep(1)  # 等待滚动完成


population_element = driver.find_element(By.XPATH, '''//*[@id="J-lemma-main-wrapper"]/div[2]/div/div[1]/div/div[5]/dl[2]/div[2]/dd/span[1]''')  # 查找包含“人口”文本的元素
driver.find_element(B)

# 获取人口数量信息
population_info = population_element.text
print(f"人口信息: {population_info}")

# 关闭浏览器
driver.quit()
