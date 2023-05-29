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
from typing import Union, Tuple, List

from InstructorEmbedding import INSTRUCTOR

def config():       
    parser = argparse.ArgumentParser( description = 'Text to Embedding Miner' )
    parser.add_argument( '--model_name', type=str, help='Name of the Instructor Embedding model to use.', default="hkunlp/instructor-xl" )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print ( config )
    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 13, config = config )

    # --- Build the synapse ---
    class InstructorTextToEmbeddingSynapse( bittensor.TextToEmbeddingSynapse ):

        def __init__( self, config ):
            super().__init__()
            self.model = INSTRUCTOR( config.model_name )
            self.device = torch.device( config.device )
            self.model.to( self.device )

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )

        def forward( self, text: List[str] ) -> torch.FloatTensor:
            return torch.tensor( self.model.encode( text ) )

    # --- Attach the synapse to the base miner ---
    text_to_embedding_synapse = InstructorTextToEmbeddingSynapse( config )
    base_miner.axon.attach( text_to_embedding_synapse )

    # --- Run miner continually until Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )
