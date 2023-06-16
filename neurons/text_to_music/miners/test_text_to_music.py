import bittensor as bt
import requests
import base64

hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

text = "Pepe the frog enormous, surrounded by colorful sea creatures and plants, - surreal, Dreamlike, ethereal lighting, Highly detailed, Intricate, Digital painting, Artstation, Concept art, Smooth, Sharp focus, Fantasy, trending on art websites, art by magali villeneuve and jock and ashley wood and rachel lee and loish"

data = { 
  "text": text,
}

# from bytes to PIL image
req = requests.post('http://127.0.0.1:8092/TextToMusic/Forward/?hotkey={}'.format(hotkey), json=data)

music_base64 = req.text

# Decode base64 string into bytes
music_bytes = base64.b64decode(music_base64)
