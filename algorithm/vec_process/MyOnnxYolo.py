from ultralytics import YOLO

class MyOnnxYolo:
    def __init__(self, model_path):
        # 加载yolo model
        self.model = YOLO(model_path, verbose=False, task='detect')

    @staticmethod
    def __get_object(img, boxes):
        if len(boxes) == 0:
            return img

        max_area = 0
        # 选出最大的框
        x1, y1, x2, y2 = 0, 0, 0, 0
        for box in boxes:
            temp_x1, temp_y1, temp_x2, temp_y2 = box
            area = (temp_x2 - temp_x1) * (temp_y2 - temp_y1)
            if area > max_area:
                max_area = area
                x1, y1, x2, y2 = temp_x1, temp_y1, temp_x2, temp_y2

        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        max_img = img[y1:y2, x1:x2, :]

        return max_img
    def get_max_box(self, img):

        results = self.model.predict(img, max_det=1, verbose=False)

        img = results[0].orig_img
        boxes = results[0].boxes.xyxy.cpu()

        max_img = self.__get_object(img, boxes)
        return max_img