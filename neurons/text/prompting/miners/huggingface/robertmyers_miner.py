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

# General.
import torch
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from base import HuggingFaceMiner

class RobertMyersMiner( HuggingFaceMiner ):

    arg_prefix: str = 'robertmyers'
    system_label: str = 'system:'
    assistant_label: str = 'assistant:'
    user_label: str = 'user:'

    def load_tokenizer( self ):
        return AutoTokenizer.from_pretrained( self.config.robertmyers.model_name )

    def load_model( self ):
        model = AutoModelForCausalLM.from_pretrained( self.config.robertmyers.model_name, torch_dtype=torch.float16 )
        model.to( self.config.robertmyers.device )
        return pipeline( 
            "text-generation", model, tokenizer=self.tokenizer, 
            device = 0, max_new_tokens = self.config.robertmyers.max_new_tokens, 
            temperature = self.config.robertmyers.temperature, 
            do_sample = self.config.robertmyers.do_sample, pad_token_id = self.tokenizer.eos_token_id 
        )

    def forward( self, messages: List[Dict[str, str]]  ) -> str:
        history = self.process_history( messages )
        resp = self.model( history )[0]['generated_text'].split(':')[-1].replace( str( history ), "")
        return resp

if __name__ == "__main__":
    bittensor.utils.version_checking()
    RobertMyersMiner().run()