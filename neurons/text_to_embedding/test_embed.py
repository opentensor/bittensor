import requests
import bittensor as bt
from io import BytesIO
import base64

bt.trace()


hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

data = {
  "text": "This is an embedding request",
  "timeout": 12,
}
req = requests.post('http://127.0.0.1:8092/TextToEmbedding/Forward/?hotkey={}'.format(hotkey), json=data)
print(req)
print(req.text)

