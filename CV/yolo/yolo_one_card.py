from Tool.MyOnnxYolo import MyOnnxYolo
import matplotlib.pyplot as plt
import PIL.Image as Image

model = MyOnnxYolo(r"D:\Code\ML\Model\Card_Seg\yolov11n_card_seg01.engine")

def show(img_path):
    model.set_result(img_path)
    model.results[0].show()

# img_path = r"D:\Code\ML\Image\_TEST_DATA\Card_test\test\2024_03_05 10_24_58.mp4-2400.jpg"
show(r"D:\Code\ML\Image\_TEST_DATA\Card_series_cls\2023-24\2023-24 CROWN ROYALE\3-2023-24 PANINI SELECT SPARKS RELICS.jpg")
print()
# img = model.get_max_img(cls_id=0)

# plt.imshow(img)
# plt.show()
# print()
