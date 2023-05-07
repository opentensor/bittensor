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

import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

class Gpt4_x_vicunaMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--gpt4_x_vicuna.model_name', type=str, required=True, help='Name/path of model to load' )
        parser.add_argument( '--gpt4_x_vicuna.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--gpt4_x_vicuna.max_new_tokens', type=int, help='Max tokens for model output.', default=256 ) 
        parser.add_argument( '--gpt4_x_vicuna.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        parser.add_argument( '--gpt4_x_vicuna.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
        parser.add_argument( '--gpt4_x_vicuna.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( '--gpt4_x_vicuna.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. " )

    def __init__( self ):
        super( Gpt4_x_vicunaMiner, self ).__init__()
        print ( self.config )
        
        bittensor.logging.info( 'Loading ' + str(self.config.gpt4_x_vicuna.model_name))
        self.tokenizer = AutoTokenizer.from_pretrained( self.config.gpt4_x_vicuna.model_name, use_fast=False )
        self.model = AutoModelForCausalLM.from_pretrained( self.config.gpt4_x_vicuna.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True )
        bittensor.logging.info( 'Model loaded!' )

        if self.config.gpt4_x_vicuna.device != "cpu":
            self.model = self.model.to( self.config.gpt4_x_vicuna.device )

    def _process_history(self, history: List[str]) -> str:
        processed_history = '### Instruction:\nWrite a good response to the last message in the conversation between you, the helpful assistant, and the user, while strictly abiding to any system prompt given.\n\nInput:\n'

        if self.config.gpt4_x_vicuna.do_prompt_injection:
            processed_history += self.config.gpt4_x_vicuna.system_prompt

        for message in history:
            if message['role'] == 'system':
                if not self.config.gpt4_x_vicuna.do_prompt_injection or message != history[0]:
                    processed_history += '\nSystem' + message['content'].strip() + ''

            if message['role'] == 'Assistant':
                processed_history += '\nAssistant:' + message['content'].strip() + ''
            if message['role'] == 'user':
                processed_history += '\nUser: ' + message['content'].strip() + ''
        return processed_history

    def forward(self, messages: List[Dict[str, str]]) -> str:

        history = self._process_history(messages)
        prompt = history + "\n### Response:"

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.gpt4_x_vicuna.device)

        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.gpt4_x_vicuna.max_new_tokens,
            temperature=self.config.gpt4_x_vicuna.temperature,
            do_sample=self.config.gpt4_x_vicuna.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Logging input and generation if debugging is active
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation))
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    Gpt4_x_vicunaMiner().run()