import base64
import requests
from io import BytesIO

hotkey = 'asdasd'
text = "My name is Terence McKenna"
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