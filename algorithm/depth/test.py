import numpy as np
from transformers import pipeline
from PIL import Image
import cv2

pipe = pipeline(task="depth-estimation", model=r"C:\Code\ML\Model\huggingface\depth-anything-small-hf")
img = Image.open(r"C:\Code\ML\Image\test02\22 (6).jpg")
depth = pipe(img)['depth']
depth.show()
