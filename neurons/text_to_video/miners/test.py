import bittensor as bt

# from bytes to PIL image
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64

hotkey = 'asdasd'

text = 'a dog running on a beach, where he encounters a crab. the dog is scared of the crab and runs away.'
seconds = 10
fps = 8

data = { 
  "text": text,
  "timeout": 120,
  "num_inference_steps": 30,
  "frames": (seconds*fps),
  "fps": fps,
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