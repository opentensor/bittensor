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
import base64
from io import BytesIO

from modelscope.pipelines import pipeline
from modelscope.outputs import OutputKeys

def config():       
    parser = argparse.ArgumentParser( description = 'Template Embdding miner.' )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print ( config )
    
    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = config.netuid, config = config )
    p = pipeline('text-to-video-synthesis', 'damo/text-to-video-synthesis')

    # --- Build the synapse ---
    class ModelscopeTextToVideo( bittensor.TextToVideoSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )
        
        def forward( self, text: List[str] ) -> Union[ torch.FloatTensor, List[float] ]:
            test_text = {
                'text': text,
            }
            output_file = p(test_text)[[OutputKeys.OUTPUT_VIDEO]]
            buffered = BytesIO()
            with open(output_file, 'rb') as f:
                encoded_string = base64.b64encode(f.read())


    # --- Attach the synapse to the base miner ---
    base_miner.attach( ModelscopeTextToVideo() )

    # --- Run the miner continually until a Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )




