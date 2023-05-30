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

import torch
import bittensor
from typing import List, Dict
from transformers import pipeline


class Dolly12BMiner( bittensor.HuggingFaceMiner ):

    arg_prefix: str = "dolly"
    assistant_label: str = "### Response:"
    user_label: str = "### Instruction:"
    system_label: str = ""

    def load_model( self ):
        bittensor.logging.info( 'Loading ' + str( self.config.dolly.model_name ) )
        model = pipeline( model=self.config.dolly.model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device=0 )
        bittensor.logging.info( 'Model loaded!' )
        return model

    def load_tokenizer( self ):
        pass

    def forward(self, messages: List[Dict[str, str]]) -> str:

        history = self.process_history( messages )
        prompt = history + self.assistant_label
        generation = self.model( prompt )

        bittensor.logging.debug(" Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    Dolly12BMiner().run()
