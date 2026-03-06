from ultralytics import YOLO

# Initialize model
model = YOLO(r"C:\Code\ML\Model\Card_Seg\yoloe-26l-seg.pt")  # or select yoloe-26s/m-seg.pt for different sizes

# Set text prompt to detect person and bus. You only need to do this once after you load the model.
model.set_classes(["text", "ball"])

# Run detection on the given image
results = model.predict(r"C:\Users\wow38\Pictures\36c2ddcc-a8bb-46b4-9aff-3a5e1ea50e25.jpg")

# Show results
results[0].show()