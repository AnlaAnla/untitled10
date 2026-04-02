from ultralytics import YOLO
from rapidocr import RapidOCR
from PIL import Image
import difflib

def ocr_card_bottom_left(image_path, crop_ratio=(0.5, 0.2)):
    """
    使用YOLO检测卡片, 裁剪卡片左下角, 并用RapidOCR识别文字.

    参数:
        yolo_model_path: YOLO模型路径
        image_path: 图片路径
        crop_ratio: 裁剪比例 (宽, 高), 左下角区域占原卡片宽高的比例

    返回:
        识别到的文字 (字符串)
    """
    # 预测
    results = yolo_model.predict(image_path)
    if len(results[0].boxes) == 0:
        print("未检测到卡片")
        return ""

    # 取第一张卡片
    box = results[0].boxes.xyxy[0].tolist()  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)

    # 打开原图
    img = Image.open(image_path)

    # 计算左下角裁剪区域
    width = x2 - x1
    height = y2 - y1
    crop_w = int(width * crop_ratio[0])
    crop_h = int(height * crop_ratio[1])

    # 左下角坐标
    crop_box = (x1, y2 - crop_h, x1 + crop_w, y2)
    cropped_img = img.crop(crop_box)

    # OCR识别
    text = ocr_engine(cropped_img).txts

    # 如果RapidOCR返回的是列表，可以拼接成字符串
    return " ".join([t[1] if isinstance(t, tuple) else t for t in text])


if __name__ == '__main__':
    # 使用示例
    yolo_model_path = r"C:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.pt"
    # 初始化模型和OCR
    yolo_model = YOLO(yolo_model_path)
    ocr_engine = RapidOCR()

    img_paths = [
        r"C:\Code\ML\Image\_TEST_DATA\Card_test\test05\945e0cc0884c8766a5883ea9593def9d.png",
        r"C:\Code\ML\Image\_TEST_DATA\Card_test\test05\65a2343f60f0115a639bf03a162d9ab8.png",
        r"C:\Code\ML\Image\_TEST_DATA\Card_test\test05\dfdb29923d90bfc254903d940fa1fd83.png"
    ]

    text_list = []
    for image_path in img_paths:
        text = ocr_card_bottom_left(image_path)
        text_list.append(text)
        print("识别结果:", text)
    print()

    for text in text_list:
        for text_str in text.split(" "):
            if '/' in text_str:
                print(text_str)

