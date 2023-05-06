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

from typing import List, Dict, Union, Tuple, Optional
from transformers import pipeline

def config():       
    parser = argparse.ArgumentParser( description = 'Whisper Text to Speech Miner' )
    parser.add_argument( '--model_name', type=str, help='Name of the speech to text model to use.', default="openai/whisper-large" )
    parser.add_argument( '--chunk_length_s', type=int, help='Audio chunk length in seconds', default=30 )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda:0" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print ( config )
    config.netuod = 13

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 13, config = config )

    # --- Build speech recognition pipeline ---
    pipe = pipeline(
        "automatic-speech-recognition",
        model = config.model_name,
        chunk_length_s = config.chunk_length_s,
        device = config.device,
    )

    # --- Build the synapse ---
    class SpeechToTextSynapse( bittensor.SpeechToTextSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )
        
        def forward( self, speech: bytes ) -> str:
            prediction = pipe( speech, return_timestamps = True )
            if isinstance(prediction, dict):
                if 'text' in prediction.keys():
                    return prediction['text']
            elif isinstance( prediction, list ):
                processor = pipe.tokenizer
                transcriptions = []
                for chunk in prediction:
                    predicted_ids = torch.tensor(chunk["token_ids"]).unsqueeze(0).to( config.stt.device )
                    transcription = processor.batch_decode( predicted_ids, skip_special_tokens=True )
                    transcriptions.append( transcription[0] )
                return transcriptions

    text_to_speech_synapse = SpeechToTextSynapse()
    base_miner.axon.attach( text_to_speech_synapse )
    base_miner.run()

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )




