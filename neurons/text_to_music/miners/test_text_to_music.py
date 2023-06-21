import bittensor as bt
import requests
import base64

hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

text = "Indian raga music - surreal, Dreamlike, ethereal, uplifting, detailed"

data = { 
  "text": text,
  "duration": 3
}

# from bytes to PIL image
req = requests.post('http://127.0.0.1:8092/TextToMusic/Forward/?hotkey={}'.format(hotkey), json=data)

music_base64 = req.text

# Decode base64 string into bytes
music_bytes = base64.b64decode(music_base64)

def decode_base64_to_wav(audio_base64: str, output_filename: str):
    audio_bytes = base64.b64decode(audio_base64)
    with open(output_filename, 'wb') as f:
        f.write(audio_bytes)
    print(f"Wav file has been written to {output_filename}")

# Now call the function
decode_base64_to_wav(music_base64, "output.wav")