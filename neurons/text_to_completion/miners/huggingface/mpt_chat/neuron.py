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
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

class Mpt_chatMiner( bittensor.HuggingFaceMiner ):
    arg_prefix: str = 'mpt_chat'
    system_label: str = '<|im_start|>system\n'
    user_label: str = '<|im_start|>user\n'
    assistant_label: str = '<|im_start|>assistant\n'

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--mpt_chat.tokenizer_name', type=str, required=False, help='Name/path of model to load' , default="EleutherAI/gpt-neox-20b")
        parser.add_argument( '--mpt_chat.use_triton', action='store_true', default=False, help='Whether to use a triton to speed up inference' )

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained( self.config.mpt_chat.tokenizer_name )

    def load_model(self):
        config = AutoConfig.from_pretrained( 'mosaicml/mpt-7b-chat', trust_remote_code=True )

        if self.config.mpt_chat.use_triton:
            config.attn_config['attn_impl'] = 'triton'

        model = AutoModelForCausalLM.from_pretrained( 
            self.config.mpt_chat.model_name, 
            torch_dtype = torch.float16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True,
            config=config
        )
        return model

    def forward(self, messages: List[Dict[str, str]]) -> str:

        history = self.process_history( messages )
        prompt = history + self.assistant_label

        input_ids = self.tokenizer.encode( prompt, return_tensors="pt" ).to( self.config.mpt_chat.device )

        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.mpt_chat.max_new_tokens,
            temperature=self.config.mpt_chat.temperature,
            do_sample=self.config.mpt_chat.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generation = self.tokenizer.decode( output[0][input_ids.shape[1]:], skip_special_tokens=False ).strip()
        generation = generation.split( "<|endoftext|>" )[0]
        
        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Prompt: " + str( prompt ) ) 
        bittensor.logging.debug( "Generation: " + str( generation.replace( "<", "-" ) ) )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    Mpt_chatMiner().run()