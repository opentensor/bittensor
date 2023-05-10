# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import time
import torch
import argparse
import bittensor

import base64
from io import BytesIO

from typing import List, Dict, Union, Tuple, Optional
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import wave
import soundfile as sf
from datasets import load_dataset



def config():       
    parser = argparse.ArgumentParser( description = 'SpeechT5 Miner' )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    bittensor.trace()
    print ( config )
    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 13, config = config )

    # --- Build speech recognition pipeline ---
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)




    # --- Build the synapse ---
    class TextToSpeechSynapse( bittensor.TextToSpeechSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return 0.0

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return False
        
        def forward( self, text: str ) -> str:
            inputs = processor(text=text, return_tensors="pt", padding=True)

            output = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
            

            audio_buffer = BytesIO()
            # with wave.open(audio_buffer, 'wb') as wav_file:
            #     # Set the parameters required for a WAV file
            #     wav_file.setnchannels(1)  # Assuming it's a mono audio
            #     wav_file.setsampwidth(2)  # Assuming 16-bit samples
            #     wav_file.setframerate(22050)
            #     wav_file.setnframes(output.numel())
            #     wav_file.writeframes(output.numpy().tobytes())

            sf.write(audio_buffer, output.numpy(), 22050, format='WAV')
            vibes = audio_buffer.getvalue()
            print (vibes)
            # Convert buffer to base64 string
            audio_base64 = base64.b64encode(vibes).decode('utf-8')
            return audio_base64
        
    # --- Attach the synapse to the base miner ---
    text_to_speech_synapse = TextToSpeechSynapse()
    base_miner.axon.attach( text_to_speech_synapse )

    # --- Run miner continually until Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )




