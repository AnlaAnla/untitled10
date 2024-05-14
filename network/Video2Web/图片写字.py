import cv2
import numpy as np

# 假设你已经有了图像帧frame,尺寸为(640, 480)
frame = cv2.imread(r"C:\Code\ML\Project\untitled10\11.jpg")
frame = cv2.resize(frame, (640, 480))


def put_text(frame, text):
    # 文本内容
    text = 'H你好ld!'

    # 选择文本字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 获取文本框的尺寸
    text_size, _ = cv2.getTextSize(text, font, 1, 2)

    # 设置文本颜色和背景颜色
    text_color = (255, 255, 255)  # 白色
    bg_color = (0, 0, 0)  # 黑色

    # 设置文本框的位置
    text_x = 10
    text_y = 40

    # 在图像上绘制背景矩形
    bg_start_x = text_x
    bg_start_y = text_y - text_size[1] - 5
    bg_end_x = bg_start_x + text_size[0] + 10
    bg_end_y = text_y + 5
    cv2.rectangle(frame, (bg_start_x, bg_start_y), (bg_end_x, bg_end_y), bg_color, -1)

    # 在图像上绘制文本
    cv2.putText(frame, text, (text_x, text_y), font, 1, text_color, 2)
    return frame


frame = put_text(frame, 'hhhhhhhh')
# 显示结果
cv2.imshow('Text on Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
