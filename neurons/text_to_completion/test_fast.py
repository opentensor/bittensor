import json
import torch
import requests
import bittensor as bt
bt.trace()

hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

data = {
  "text": text,
  "timeout": 12,
}
data = {
    "roles": ["system", "user"],
    "messages": ["You are an unhelpful assistant.", "What is the capital of Texas?"],
    "timeout": 12,
}

req = requests.post('http://127.0.0.1:8092/TextToCompletion/Forward/?hotkey={}'.format(hotkey), json=data)
print(req)
print(req.text)