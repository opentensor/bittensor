import json
import torch
import requests
import bittensor as bt
bt.trace()

hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'
text = "(beautiful) (best quality) mdjrny-v4 style Pepe the frog enormous, surrounded by colorful sea creatures and plants, - surreal, Dreamlike, ethereal lighting, Highly detailed, Intricate, Digital painting, Artstation, Concept art, Smooth, Sharp focus, Fantasy, trending on art websites, art by magali villeneuve and jock and ashley wood and rachel lee and loish"
texts = ["This is a list of strings", "A second string"]

data = {
  "text": text,
  "timeout": 12,
}
req = requests.post('http://127.0.0.1:8092/TextToEmbedding/Forward/?hotkey={}'.format(hotkey), json=data)
print(req)
print(req.text)
x = json.loads(req.text)
emb = torch.Tensor(x)
print("emb.shape: ", emb.shape)

# TODO: Fix the proto so we can send a list of strings (repeated like text_propmting)
data = {
  "text": json.dumps(texts),
  "timeout": 12,
}
req = requests.post('http://127.0.0.1:8092/TextToEmbedding/Forward/?hotkey={}'.format(hotkey), json=data)
print(req)
print(req.text)
x = json.loads(req.text)
emb = torch.Tensor(x)
print("emb.shape: ", emb.shape)