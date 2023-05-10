import bittensor as bt

# from bytes to PIL image
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64

hotkey = 'asdasd'

text = 'hello world'

data = { 
  "text": "a dog in the forest",
  "height": 256,
  "width": 256,
  "timeout": 12,
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