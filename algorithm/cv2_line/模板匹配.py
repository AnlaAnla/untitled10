import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

img_dir_path = r"C:\Code\ML\Image\Card_test\match_test"
template_img_paths = [r"C:\Code\ML\Image\Card_test\match_tamplate\corner1.png",
                      r"C:\Code\ML\Image\Card_test\match_tamplate\corner2.png",
                      r"C:\Code\ML\Image\Card_test\match_tamplate\corner3.png",
                      r"C:\Code\ML\Image\Card_test\match_tamplate\corner4.png"
                      # r"C:\Code\ML\Image\Card_test\test03\margin_corner1.jpg",
                      # r"C:\Code\ML\Image\Card_test\test03\margin_corner2.jpg",
                      # r"C:\Code\ML\Image\Card_test\test03\margin_corner3.jpg",
                      # r"C:\Code\ML\Image\Card_test\test03\margin_corner4.jpg"
                      ]


# 获取模板的宽度和高度


def match_loc(target_img_path, template_img_path):
    img = cv2.imread(target_img_path)
    # 读取目标图像和模板图像
    template_img = cv2.imread(template_img_path, cv2.IMREAD_GRAYSCALE)
    target_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    w, h = template_img.shape[::-1]

    # 使用cv2.matchTemplate进行模板匹配
    result = cv2.matchTemplate(target_img, template_img, cv2.TM_CCOEFF_NORMED)

    # 设置相似度阈值（例如0.3，较低的阈值）
    # threshold = 0.6
    # loc = np.where(result >= threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    # 在目标图像上绘制匹配结果

    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    return [top_left, bottom_right]


def match(img_path):
    locations = []
    for template_img_path in template_img_paths:
        locations.append(match_loc(img_path, template_img_path))
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

    point = [[locations[0][0][0], locations[0][0][1]], [locations[1][1][0], locations[1][0][1]],
             [locations[3][1][0], locations[3][1][1]], [locations[2][0][0], locations[2][1][1]]]
    points = np.array(point)
    cv2.polylines(img, [points], True, (0, 255, 0), 2)
    # for top_left, bottom_right in locations:
    #     point.append(top_left)
    #     cv2.rectangle(img, top_left, bottom_right, (250, 0, 0), 2)

    plt.figure()
    plt.imshow(img)
    plt.show()


# for img_name in os.listdir(img_dir_path):
#     img_path = os.path.join(img_dir_path, img_name)
#     match(img_path)

match(r"C:\Code\ML\Image\Card_test\match_test\2024_06_19___06_18_46.jpg")
