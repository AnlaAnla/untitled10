from ultralytics import SAM
import matplotlib.pyplot as plt

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a point prompt
result = model.predict(r"C:\Users\wow38\Pictures\Screenshots\屏幕截图 2024-06-03 104446.png", points=[900, 370], labels=[1])

print(result)