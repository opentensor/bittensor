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
import torch
import argparse
import bittensor

import base64
from io import BytesIO
from typing import Union, Tuple, List
import soundfile as sf
from datasets import load_dataset
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration


def config():       
    parser = argparse.ArgumentParser( description = 'Librispeech Miner' )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    bittensor.trace()
    print ( config )
    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 13, config = config )

    # --- Build speech recognition pipeline ---
    processor = Speech2TextProcessor.from_pretrained( "facebook/s2t-small-librispeech-asr" )
    model = Speech2TextForConditionalGeneration.from_pretrained( "facebook/s2t-small-librispeech-asr" )


    # --- Build the synapse ---
    class SpeechToTextSynapse( bittensor.SpeechToTextSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return 0.0

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return False
        
        def forward( self, speech: bytes ) -> str:
            # Decode the base64 string to bytes.
            audio_bytes = base64.b64decode( speech )

            # Convert bytes to numpy array.
            audio_array, samplerate = sf.read( BytesIO( audio_bytes ) )

            # Process the decoded input audio.
            inputs = processor( audio_array, sampling_rate=samplerate, return_tensors="pt" )
            generated_ids = model.generate( inputs[ "input_features" ], attention_mask=inputs[ "attention_mask" ] )

            # Generate the transcription.
            return processor.decode( generated_ids[ 0 ], skip_special_tokens=True )
        
    # --- Attach the synapse to the base miner ---
    speech_to_text_synapse = SpeechToTextSynapse()
    base_miner.axon.attach( speech_to_text_synapse )

    # --- Run miner continually until Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )
