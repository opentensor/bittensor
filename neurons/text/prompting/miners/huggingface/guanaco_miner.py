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
from transformers import AutoTokenizer, AutoModelForCausalLM

from base import HuggingFaceMiner

class GuanacoMiner( HuggingFaceMiner ):

    arg_prefix: str = 'guanaco'
    assistant_label: str = '### Assistant:'
    user_label: str = '### Human:'
    system_label: str = ''

    def load_tokenizer( self ):
        return AutoTokenizer.from_pretrained( self.config.guanaco.model_name )

    def load_model( self ):
        return AutoModelForCausalLM.from_pretrained(
            self.config.guanaco.model_name, 
            torch_dtype = torch.float16, 
            low_cpu_mem_usage=True, 
            device_map=self.config.guanaco.device_map 
        )

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history( messages )
        prompt = history + self.assistant_label

        generate_kwargs = dict(
            temperature=self.config.guanaco.temperature,
            max_new_tokens=self.config.guanaco.max_new_tokens,
            top_p=self.config.guanaco.top_p,
            repetition_penalty=self.config.guanaco.repetition_penalty,
            do_sample=self.config.guanaco.do_sample,
        )
        if '33B' in self.config.guanaco.model_name: # Tim Dettmers 33B model-specific parameters
            generate_kwargs['truncate'] = 999
            generate_kwargs['seed'] = 42

        input_ids = self.tokenizer.encode( prompt, return_tensors="pt" ).to( self.config.guanaco.device )
        output = self.model.generate(
            input_ids,
            **generate_kwargs
        )
        generated_text = self.tokenizer.decode( output[0][input_ids.shape[1]:], skip_special_tokens=True )
        generation = generated_text.split( self.assistant_label )[0].strip()

        bittensor.logging.debug("Message: " + str( messages ) )
        bittensor.logging.debug("Generation: " + str( generation ) )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    GuanacoMiner().run()