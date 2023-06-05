# The MIT License (MIT)
# Copyright © 2023 Opentensor Foundation

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
from transformers import AutoTokenizer, AutoModelForCausalLM


class NeoxtMiner( bittensor.HuggingFaceMiner ):
    arg_prefix: str = 'neoxt'
    assistant_label: str = '<bot>:'
    user_label: str = '<human>:'
    system_label: str = ''

    def load_tokenizer( self ):
        return AutoTokenizer.from_pretrained( self.config.neoxt.model_name )

    def load_model( self ):
        return AutoModelForCausalLM.from_pretrained( self.config.neoxt.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True )

    def forward( self, messages: List[Dict[str, str]] ) -> str:
        history = self.process_history( messages )
        prompt = history + self.assistant_label
        input_ids = self.tokenizer.encode( prompt, return_tensors="pt" ).to( self.config.neoxt.device )
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.neoxt.max_new_tokens,
            temperature=self.config.neoxt.temperature,
            do_sample=self.config.neoxt.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated_text = self.tokenizer.decode( output[0][input_ids.shape[1]:], skip_special_tokens=True )
        generation = generated_text.split( "<human>" )[0].strip()

        bittensor.logging.debug( "Message: " + str( messages ).replace( "<","-" ).replace( ">","-" ) )
        bittensor.logging.debug( "Generation: " + str( generation ).replace( "<","-" ).replace( ">","-" ) )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    NeoxtMiner().run()
