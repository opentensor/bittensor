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
  "text": "a dog running in the redwood forest",
  "height": 512, # anything less than 512x512 causes image degradation
  "width": 512,
  "timeout": 12,
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "negative_prompt": ""
}

# from bytes to PIL image
req = requests.post('http://127.0.0.1:8092/TextToVideo/Forward/?hotkey={}'.format(hotkey), json=data)

vid_base64 = req.text

# Decode base64 string into bytes
vid_bytes = base64.b64decode(vid_base64)

# Load bytes into a PIL image
with open('test.mp4', 'wb') as f:
    f.write(vid_bytes)

print('saved video to test.png')