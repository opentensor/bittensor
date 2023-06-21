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

import os
import time
import argparse
import bittensor
import base64
from io import BytesIO

import soundfile as sf
from audiocraft.models import musicgen
from audiocraft.data.audio import audio_write

from typing import List

bittensor.trace()

class AudioCraftTextToMusicMiner( bittensor.BaseTextToMusicMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--neuron.model_name', type=str, help='Name of the model to use.', default = "medium" )
        parser.add_argument( '--tempfile', type=str, default='tempfile')
        parser.add_argument( '--device', type=str, help='Device to load model', default="cuda" )

    def __init__( self, config: "bittensor.Config" = None ):
        super( AudioCraftTextToMusicMiner, self ).__init__( config = config )
        self.model = musicgen.MusicGen.get_pretrained(self.config.neuron.model_name, device=self.config.device)
    
    def forward( self, text: str, duration: int, ) -> List[str]:
        # Perform the generation
        self.model.set_generation_params( duration=duration )
        output = self.model.generate( [text], progress=True )
        # Convert to audio using Audiocraft tool and save as a tempfile
        audio_write( 
            f'{self.config.tempfile}', 
            output[0].cpu(), 
            self.model.sample_rate, 
            strategy="loudness", 
            loudness_compressor=True 
        )

        # Convert generated wav to bytes in base64
        audio_buffer = BytesIO()
        with open( f'{self.config.tempfile}.wav', 'rb' ) as f:
            audio_array, _ = sf.read(f'{self.config.tempfile}.wav')
            sf.write( audio_buffer, audio_array, self.model.sample_rate, format='WAV' )
            audio_base64 = base64.b64encode( audio_buffer.getvalue() ).decode( 'utf-8' )
            f.close()

        # Cleanup
        os.system("rm -rf {tempfile}.wav")

        return audio_base64


if __name__ == "__main__":
    bittensor.utils.version_checking()
    with AudioCraftTextToMusicMiner() as miner:
        while True:
            time.sleep( 1 )
