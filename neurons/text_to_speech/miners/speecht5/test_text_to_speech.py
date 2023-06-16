import base64
import requests
from io import BytesIO

hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'
text = 'I have been trying to get ahold of you to let you know about your car\'s extended warranty. Please press 1 to speak to a representative.'

data = {
    "text": text
}

req = requests.post('http://127.0.0.1:8092/TextToSpeech/Forward/?hotkey={}'.format(hotkey), json=data)

audio_base64 = req.text

# Decode base64 string into bytes
audio_bytes = base64.b64decode(audio_base64)
print('audio_bytes:', audio_bytes)
# Save bytes as a WAV file
with open('output.wav', 'wb') as f:
    f.write(audio_bytes)