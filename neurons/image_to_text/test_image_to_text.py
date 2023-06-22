
import requests
import base64
import numpy as np
from PIL import Image
import io

hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

def load_image_and_serialize(image_path):
    # Load image with Pillow and convert to numpy array
    image = Image.open(image_path)
    np_image = np.array(image)

    # Convert numpy array to bytes
    image_bytes = io.BytesIO()
    np.save(image_bytes, np_image, allow_pickle=True)
    image_bytes = image_bytes.getvalue()

    # Convert bytes to base64 bytestring
    base64_bytestring = base64.b64encode(image_bytes).decode('utf-8')

    return base64_bytestring

# Usagei
# image_base64 = load_image_to_base64('/home/jason/FIX/bittensor/neurons/image_to_text/onthemountain.jpeg')
image_base64 = load_image_and_serialize('neurons/image_to_text/onthemountain.jpeg')

data = { 
  "image": image_base64,
  "timeout": 12
}

# from bytes to PIL image
req = requests.post('http://127.0.0.1:8092/ImageToText/Forward/?hotkey={}'.format(hotkey), json=data)

caption = req.text

print("Caption:", caption)