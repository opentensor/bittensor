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
import soundfile as sf
from datasets import load_dataset


class SpeechT5TextToSpeechMiner( bittensor.BaseTextToSpeechMiner ):

    samplerate: int = 22050
    audio_format: str = 'WAV'

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--speecht5.model_name', default="microsoft/speecht5_tts", type=str, help='Name of the model to use for speech to text conversion' )
        parser.add_argument( '--speecht5.vocoder_name', type=str, help='Name of vocoder model', default="microsoft/speecht5_hifigan" )
        parser.add_argument( '--speecht5.device', type=str, help='Device to load model', default="cuda:0" )

    def __init__(self, config: "bittensor.Config" = None ):
        super( SpeechT5TextToSpeechMiner, self ).__init__( config=config )

        # --- Build speech recognition pipeline ---
        self.processor = SpeechT5Processor.from_pretrained( "microsoft/speecht5_tts" )
        self.model = SpeechT5ForTextToSpeech.from_pretrained( "microsoft/speecht5_tts" )
        self.vocoder = SpeechT5HifiGan.from_pretrained( "microsoft/speecht5_hifigan" )
        self.embeddings_dataset = load_dataset( "Matthijs/cmu-arctic-xvectors", split="validation" )
        self.speaker_embeddings = torch.tensor( self.embeddings_dataset[ 7306 ][ "xvector"] ).unsqueeze( 0 )

    def forward( self, text: str ) -> str:
        # --- Generate audio from text
        inputs = self.processor( text=text, return_tensors="pt", padding=True )
        output = self.model.generate_speech( inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder )
    
        # --- Convert tensor to audio buffer --- 
        audio_buffer = BytesIO()
        sf.write( audio_buffer, output.numpy(), self.samplerate, format=self.audio_format )
        vibes = audio_buffer.getvalue()

        # --- Convert buffer to base64 string ---
        audio_base64 = base64.b64encode( vibes ).decode( 'utf-8' )
        return audio_base64


if __name__ == "__main__":
    bittensor.utils.version_checking()
    # --- Run miner continually until Keyboard break ---
    with SpeechT5TextToSpeechMiner(): 
        while True: 
            time.sleep( 1 )
 


