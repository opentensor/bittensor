import bittensor as bt

# from bytes to PIL image
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64

hotkey = 'asdasd'

text = "mdjrny-v4 style Pepe the frog enormous, surrounded by colorful sea creatures and plants, - surreal, Dreamlike, ethereal lighting, Highly detailed, Intricate, Digital painting, Artstation, Concept art, Smooth, Sharp focus, Fantasy, trending on art websites, art by magali villeneuve and jock and ashley wood and rachel lee and loish"


# open image from path and be PIL.Image
image = Image.open('/home/carro/pepe.jpg')

buffered = BytesIO()
image.save(buffered, format="PNG")  # You can use "PNG" if you prefer

# Encode image buffer to base64
image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
data = { 
  "text": text,
  "image": image_base64,
  # "height": 768, # anything less than 512x512 causes image degradation
  # "width": 512,
  "timeout": 120,
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "negative_prompt": ""
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