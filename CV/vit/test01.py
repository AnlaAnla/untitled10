from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import requests
import time

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

for i in range(2):
    t1 = time.time()
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state

    print('time: ', time.time() - t1)

print()