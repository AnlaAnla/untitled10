from ultralytics import YOLO

# 初始化模型
model = YOLO(r"C:\Code\ML\Model\Card_Seg\card_hand_yoloe26s_seg01.pt")

# 预测
results = model.predict(
    r"C:\Users\wow38\Pictures\videoframe_6871967 - 副本.png"
)

# 显示结果
results[0].show()