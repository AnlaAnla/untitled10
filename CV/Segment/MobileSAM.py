from ultralytics import SAM
import matplotlib.pyplot as plt

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a point prompt
result = model.predict(r"C:\Users\wow38\Pictures\scenery\background.jpg", points=[900, 370], labels=[1])

print(result)