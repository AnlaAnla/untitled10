import cv2
import numpy as np
from ultralytics import YOLO


class CardProcessor:
    def __init__(self, yolo_path, target_size=384):
        """
        初始化处理器
        :param yolo_path: YOLO 模型路径
        :param target_size: 目标图片大小 (Swin通常用 384)
        """
        self.model = YOLO(yolo_path)
        self.target_size = target_size

    def letterbox_image(self, img, new_shape=(384, 384), color=(0, 0, 0)):
        """
        核心函数：不改变长宽比的 Resize，并进行 Padding
        """
        shape = img.shape[:2]  # current shape [height, width]

        # 计算缩放比例 (取最小比例，保证图片能完整塞进去)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # 计算 resize 后的宽高
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        # 计算需要填充多少像素 (dw, dh)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

        # 将 padding 分摊到上下左右
        dw /= 2
        dh /= 2

        # 1. Resize 图片
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        # 2. 添加边框 (Padding)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # 使用纯色填充边框 (默认黑色 0,0,0)
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)

        # 强制 Resize 到精确目标尺寸 (防止计算误差导致差 1-2 像素)
        return cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)

    def process_single_image(self, image_path, save_path=None):
        """
        读取 -> YOLO检测 -> 裁剪 -> Padding -> 返回/保存
        """
        # 1. 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: 无法读取图片 {image_path}")
            return None

        # 2. YOLO 推理
        # conf=0.25 过滤低置信度, save=False 不保存yolo自带的结果
        results = self.model(img, conf=0.25, verbose=False)

        # 3. 获取检测框
        if len(results[0].boxes) == 0:
            print(f"Warning: 未检测到卡片 {image_path}，将使用原图处理")
            crop_img = img
        else:
            # 默认取置信度最高的一个框
            box = results[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 增加一点容错，防止切得太死把卡边切掉了 (比如外扩 5 像素)
            h, w, _ = img.shape
            x1 = max(0, x1 - 5)
            y1 = max(0, y1 - 5)
            x2 = min(w, x2 + 5)
            y2 = min(h, y2 + 5)

            crop_img = img[y1:y2, x1:x2]

        # 4. Swin 适配 (Letterbox Resize)
        final_img = self.letterbox_image(crop_img, new_shape=(self.target_size, self.target_size))

        # 5. 保存或返回
        if save_path:
            cv2.imwrite(save_path, final_img)
            return True
        else:
            return final_img


# --- 使用示例 ---
if __name__ == "__main__":
    # 初始化
    processor = CardProcessor(
        yolo_path=r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt",
        target_size=384  # Swin Base 384
    )

    # 处理一张图片
    test_img = r"C:\Code\ML\Image\_CLASSIFY\card_cls\train_test\train\#1 kyle kuzma\IMG_5742.JPG"
    save_to = r"test_card_384.jpg"

    processor.process_single_image(test_img, save_to)
