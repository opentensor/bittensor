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
from transformers import AutoTokenizer, AutoModelForCausalLM

def config():       
    parser = argparse.ArgumentParser( description = 'Whisper Text to Speech Miner' )
    parser.add_argument( '--vicuna.model_name', type=str, required=True, help='Name/path of model to load' )
    parser.add_argument( '--vicuna.device', type=str, help='Device to load model', default="cuda" )
    parser.add_argument( '--vicuna.max_new_tokens', type=int, help='Max tokens for model output.', default=256 ) 
    parser.add_argument( '--vicuna.temperature', type=float, help='Sampling temperature of model', default=0.5 )
    parser.add_argument( '--vicuna.do_sample', action='store_true', default=False, help='Whether to use sampling or not (if not, uses greedy decoding).' )
    parser.add_argument( '--vicuna.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
    parser.add_argument( '--vicuna.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. " )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    print ( config )
    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 1, config = config )

    # --- Build vicuna model. ---
    bittensor.logging.info( 'Loading ' + str( config.vicuna.model_name))
    tokenizer = AutoTokenizer.from_pretrained( config.vicuna.model_name, use_fast=False )
    model = AutoModelForCausalLM.from_pretrained( config.vicuna.model_name, torch_dtype = torch.float16, low_cpu_mem_usage=True )
    if config.vicuna.device != "cpu":
        model = model.to( config.vicuna.device )
    bittensor.logging.info( 'Model loaded!' )

    # --- Process history helper ---
    def process_history( history: List[str] ) -> str:
        processed_history = ''
        if config.vicuna.do_prompt_injection:
            processed_history += config.vicuna.system_prompt
        for message in history:
            if message['role'] == 'system':
                if not config.vicuna.do_prompt_injection or message != history[0]:
                    processed_history += '' + message['content'].strip() + ' '
            if message['role'] == 'Assistant':
                processed_history += 'ASSISTANT:' + message['content'].strip() + '</s>'
            if message['role'] == 'user':
                processed_history += 'USER: ' + message['content'].strip() + ' '
        return processed_history

    # --- Build the synapse ---
    class Vicuna( bittensor.TextPromptingSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            return base_miner.priority( forward_call )

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            return base_miner.blacklist( forward_call )
        
        def forward( self, messages: List[Dict[str, str]] ) -> str:

            # Process inputs.
            history = process_history(messages)
            prompt = history + "ASSISTANT:"
            input_ids = tokenizer.encode( prompt, return_tensors = "pt" ).to( config.vicuna.device )

            # Run model
            output = model.generate(
                input_ids,
                max_length = input_ids.shape[1] + config.vicuna.max_new_tokens,
                temperature = config.vicuna.temperature,
                do_sample = config.vicuna.do_sample,
                pad_token_id = tokenizer.eos_token_id,
            )

            # Return detokenized outputs.
            generation = tokenizer.decode( output[0][input_ids.shape[1]:], skip_special_tokens = True )
            bittensor.logging.debug("Message: " + str(messages))
            bittensor.logging.debug("Generation: " + str(generation))
            return generation

    # --- Attach the synapse to the base miner ---
    vicuna = Vicuna()
    base_miner.axon.attach( vicuna )

    # --- Run the miner continually until a Keyboard break ---
    with base_miner: 
        while True: 
            time.sleep( 1 )

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )




