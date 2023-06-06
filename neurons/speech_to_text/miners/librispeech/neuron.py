# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
import argparse
import bittensor
import base64
from io import BytesIO
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf

class LibriSpeechToTextMiner( bittensor.BaseSpeechToTextMiner ):
    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--librispeech.model_name', default="facebook/s2t-small-librispeech-asr", type=str, help='Name of the model to use for speech to text conversion' )
        parser.add_argument( '--librispeech.device', type=str, help='Device to load model', default="cuda:0" )

    def __init__(self, *args, **kwargs):
        super( LibriSpeechToTextMiner, self ).__init__( *args, **kwargs )
        self.processor = Speech2TextProcessor.from_pretrained( self.config.librispeech.model_name )
        self.model = Speech2TextForConditionalGeneration.from_pretrained( 
            self.config.librispeech.model_name 
        ).to( self.config.librispeech.device )

    def forward( self, speech: bytes ) -> str:
        audio_bytes = base64.b64decode( speech )
        audio_array, samplerate = sf.read( BytesIO( audio_bytes ) )
        inputs = self.processor( audio_array, sampling_rate=samplerate, return_tensors="pt" ).to( self.config.librispeech.device )
        generated_ids = self.model.generate( inputs[ "input_features" ], attention_mask=inputs[ "attention_mask" ] )
        return self.processor.decode( generated_ids[ 0 ], skip_special_tokens=True )


if __name__ == "__main__":
    bittensor.utils.version_checking()
    with LibriSpeechToTextMiner():
        while True:
            time.sleep(1)
