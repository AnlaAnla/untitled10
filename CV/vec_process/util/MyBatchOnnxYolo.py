import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Union, Optional
import PIL.Image

ImageType = Union[str, np.ndarray, PIL.Image.Image]


class MyBatchOnnxYolo:
    """
    使用 YOLO 模型进行批处理目标检测/分割，并提供提取最大目标区域的功能。
    cls_id {card:0} - 根据你的模型调整
    """

    def __init__(self, model_path: str, task: str = 'segment', verbose: bool = False):
        # 加载yolo model
        self.model = YOLO(model_path, task=task, verbose=verbose)
        self.results: Optional[List] = None  # 将存储批处理的结果列表
        self.batch_size: int = 0

    def predict_batch(self, image_list: List[ImageType], imgsz: int = 640, **kwargs):
        """
        对一批图像进行预测。

        Args:
            image_list (List[ImageType]): 包含图像路径、PIL Image 或 NumPy 数组的列表。
            imgsz (int): 推理的图像尺寸。
            **kwargs: 其他传递给 model.predict 的参数 (例如 conf, iou)。
        """
        if not image_list:
            print("Warning: Input image list is empty.")
            self.results = []
            self.batch_size = 0
            return

        # 使用 YOLO 的批处理能力
        self.results = self.model.predict(image_list, verbose=False, imgsz=imgsz, **kwargs)
        self.batch_size = len(self.results)

    def get_batch_size(self) -> int:
        return self.batch_size

    def _get_result_at_index(self, index: int):
        """内部辅助方法，获取指定索引的结果，并进行边界检查。"""
        if self.results is None:
            raise ValueError("Must call predict_batch() before accessing results.")
        if not (0 <= index < self.batch_size):
            raise IndexError(f"Index {index} is out of bounds for batch size {self.batch_size}.")
        return self.results[index]

    def check(self, index: int, cls_id: int) -> bool:
        """
        检查指定索引的图像结果中是否存在特定的类别ID。

        Args:
            index (int): 图像在批处理中的索引 (从0开始)。
            cls_id (int): 要检查的类别ID。

        Returns:
            bool: 如果存在该类别ID，则返回 True，否则返回 False。
        """
        result = self._get_result_at_index(index)
        if result.boxes is None or len(result.boxes) == 0:
            return False
        # .cls 可能为空 Tensor，需要检查
        return result.boxes.cls is not None and cls_id in result.boxes.cls.cpu().tolist()

    def get_max_img(self, index: int, cls_id: int = 0) -> Optional[np.ndarray]:
        """
        从指定索引的图像结果中，提取指定类别ID的最大边界框对应的图像区域。

        Args:
            index (int): 图像在批处理中的索引 (从0开始)。
            cls_id (int): 要提取的目标类别ID 默认0

        Returns:
            Optional[np.ndarray]: 裁剪出的最大目标的图像区域 (RGB NumPy 数组)，
                                   如果未找到该类别或无检测结果，则返回原始图像。
        """
        result = self._get_result_at_index(index)
        orig_img = result.orig_img  # 通常是 BGR NumPy 数组
        boxes = result.boxes

        # 检查是否有检测框以及是否有对应的类别
        if boxes is None or len(boxes) == 0 or boxes.cls is None or cls_id not in boxes.cls.cpu():
            print(
                f"Warning: No detections or cls_id {cls_id} not found for image at index {index}. Returning original image.")
            # 返回原始图像的 RGB 版本
            return cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) if orig_img is not None else None

        max_area = 0.0
        max_box = None

        xyxy_boxes = boxes.xyxy.cpu().numpy()
        cls_list = boxes.cls.cpu().numpy()

        # 选出最大的目标框
        for i, box in enumerate(xyxy_boxes):
            if cls_list[i] != cls_id:
                continue

            temp_x1, temp_y1, temp_x2, temp_y2 = box
            area = (temp_x2 - temp_x1) * (temp_y2 - temp_y1)
            if area > max_area:
                max_area = area
                max_box = box

        # 如果没有找到对应 cls_id 的框 (理论上前面已检查，但多一层保险)
        if max_box is None:
            print(
                f"Warning: cls_id {cls_id} found in cls_list but failed to find max box for image at index {index}. Returning original image.")
            return cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) if orig_img is not None else None

        x1, y1, x2, y2 = map(int, max_box)  # 转换为整数坐标

        # 边界处理，防止裁剪坐标超出图像范围
        h, w = orig_img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        # 检查裁剪区域是否有效
        if x1 >= x2 or y1 >= y2:
            print(
                f"Warning: Invalid crop dimensions [{y1}:{y2}, {x1}:{x2}] for image at index {index}. Returning original image.")
            return cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB) if orig_img is not None else None

        # 裁剪图像 (orig_img 通常是 BGR)
        max_img_crop = orig_img[y1:y2, x1:x2]

        # 将裁剪结果转换为 RGB (与 matplotlib 和 PIL 更兼容)
        max_img_rgb = cv2.cvtColor(max_img_crop, cv2.COLOR_BGR2RGB)

        return max_img_rgb

    def get_max_img_list(self, cls_id: int = 0) -> List[Optional[np.ndarray]]:
        """
        对批处理中的每张图片，提取指定类别ID的最大边界框对应的图像区域。

        Args:
            cls_id (int): 要提取的目标类别ID 默认0

        Returns:
            List[Optional[np.ndarray]]: 包含处理后图像 (RGB NumPy 数组) 的列表。
                                        对于成功裁剪的图片，列表元素是裁剪后的图像。
                                        如果某张图片未找到指定类别或裁剪失败，列表元素是该图片的原始图像(RGB)。
                                        如果原始图像无效，则列表元素为 None。
        """
        if self.results is None:
            raise ValueError("Must call predict_batch() before calling get_max_img_list().")

        processed_images: List[Optional[np.ndarray]] = []
        for i in range(self.batch_size):
            # 调用 get_max_img 获取单张图片的处理结果
            processed_img = self.get_max_img(index=i, cls_id=cls_id)
            processed_images.append(processed_img)

        return processed_images

    def get_results(self) -> Optional[List]:
        """获取完整的批处理结果列表。"""
        return self.results
