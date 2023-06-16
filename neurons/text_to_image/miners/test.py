import bittensor as bt

# from bytes to PIL image
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64
import torch

hotkey = 'asdasd'


text = "(beautiful) (best quality) A gorgeous woman at the beach, Portrait Photograph, Highly detailed, Intricate, Digital painting, Artstation, Concept art, Smooth, Sharp focus, Realistic, art by magali villeneuve and jock and ashley wood and rachel lee and loish"
negative_prompt = "(worst quality) (poorly drawn) "
buffered = BytesIO()
seed = torch.randint(1000000000, (1,)).item()


data = { 
  "text": text,
  "height": 768, # anything less than 512x512 causes image degradation
  "width": 512,
  "timeout": 120,
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "strength": 0.75,
  "negative_prompt": negative_prompt,
  "seed": seed
}

# from bytes to PIL image
req = requests.post('http://127.0.0.1:8092/TextToImage/Forward/?hotkey={}'.format(hotkey), json=data)

img_base64 = req.text

# Decode base64 string into bytes
img_bytes = base64.b64decode(img_base64)

# Load bytes into a PIL image
pil = Image.open(BytesIO(img_bytes))

pil.save('test.png')
print('saved image to test.png')

print("seed: ", seed)

# high strength (less like input image)
data_img_2_img = {
  "text": text,
  "image": img_base64,
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "strength": 0.9,
  "negative_prompt": negative_prompt,
  "seed": seed
}

req = requests.post('http://127.0.0.1:8092/TextToImage/Forward/?hotkey={}'.format(hotkey), json=data_img_2_img)

img_base64_2 = req.text

# Decode base64 string into bytes

img_bytes = base64.b64decode(img_base64_2)

# Load bytes into a PIL image
pil = Image.open(BytesIO(img_bytes))

pil.save('test2.png')
print('saved image to test2.png')

# same with strength of 0.5 (more like input image)
data_img_2_img = {
  "text": text,
  "image": img_base64,
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "strength": 0.5,
  "negative_prompt": negative_prompt,
  "seed": seed
}

req = requests.post('http://127.0.0.1:8092/TextToImage/Forward/?hotkey={}'.format(hotkey), json=data_img_2_img)

img_base64_3 = req.text

# Decode base64 string into bytes

img_bytes = base64.b64decode(img_base64_3)

# Load bytes into a PIL image
pil = Image.open(BytesIO(img_bytes))

pil.save('test3.png')
print('saved image to test3.png')

