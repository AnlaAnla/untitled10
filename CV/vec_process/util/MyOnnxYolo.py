from ultralytics import YOLO


class MyOnnxYolo:
    # cls_id {card:0, person:1, hand:2}
    def __init__(self, model_path):

        # 加载yolo model

        self.model = YOLO(model_path, verbose=False, task='detect')
        self.results = None

        self.cls = None
        self.boxes = None
        self.img = None

    def set_result(self, img, imgsz=640):
        self.results = self.model.predict(img, max_det=3, verbose=False, imgsz=imgsz)

        self.img = self.results[0].orig_img
        self.boxes = self.results[0].boxes.xyxy.cpu()
        self.cls = self.results[0].boxes.cls

    def check(self, cls_id: int):
        if cls_id in self.cls:
            return True
        return False

    def get_max_img(self, cls_id: int):
        # cls_id {card:0, person:1, hand:2}
        # 排除没有检测到物体 或 截取的id不存在的图片
        if len(self.boxes) == 0 or cls_id not in self.cls:
            return self.img

        max_area = 0
        # 选出最大的卡片框
        x1, y1, x2, y2 = 0, 0, 0, 0
        for i, box in enumerate(self.boxes):
            if self.cls[i] != cls_id:
                continue

            temp_x1, temp_y1, temp_x2, temp_y2 = box
            area = (temp_x2 - temp_x1) * (temp_y2 - temp_y1)
            if area > max_area:
                max_area = area
                x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        max_img = self.img[y1:y2, x1:x2, :]

        return max_img
