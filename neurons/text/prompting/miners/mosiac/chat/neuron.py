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
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers import StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids: List[int] = None):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class Mpt7BMiner( bittensor.BasePromptingMiner ):

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--mpt7B.model_name', type=str, default='mosaicml/mpt-7b-chat', help='Name/path of model to load' )
        parser.add_argument( '--mpt7B.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--mpt7B.max_new_tokens', type=int, help='Max tokens for model output.', default=256 )
        parser.add_argument( '--mpt7B.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        parser.add_argument( '--mpt7B.greedy_sampling', action='store_true', default=False, help='Whether to use greedy sampling or not (if not, uses multinomial sampling).')
        parser.add_argument( '--mpt7B.repetition-penalty', type=float, default=1.1, help='Repetition penalty for greedy decoding. Between 1.0 and infinity. 1.0 means no penalty. Default: 1.0' )
        parser.add_argument( '--mpt7B.top_p', type=float, default=0.9, help='Top-p (nucleus) sampling. Defaults to 1.0 (top-k sampling). Must be between 0.0 and 1.0.' )
        parser.add_argument( '--mpt7B.top_k', type=int, default=0, help='Top-k sampling. Defaults to 0 (no top-k sampling). Must be between 0 and 1000.' )


    def __init__( self ):
        super( Mpt7BMiner, self ).__init__()
        print ( self.config )

        self.system_prompt = self.config.mpt7B.system_prompt
        self.system_key = "### System: "
        self.assistant_key = "### Response: "
        self.user_key = "### Instruction: "

        bittensor.logging.info( 'Loading ' + str(self.config.mpt7B.model_name))
        model_config = AutoConfig.from_pretrained(
            self.config.mpt7B.model_name,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.mpt7B.model_name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained( "EleutherAI/gpt-neox-20b" )
        self.stop = StopOnTokens( self.tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]) )

        bittensor.logging.info( 'Model loaded!' )

        if self.config.mpt7B.device != "cpu":
            self.model = self.model.to( self.config.mpt7B.device )

    def _process_history( self, history: List[Dict[str, str]] ) -> str:
        processed_history = ''

        for message in history:
            if message['role'].lower() == 'system':
                if message != history[0]:
                    processed_history += self.system_key + message['content'].strip() + ' '
            if message['role'].lower() == 'assistant':
                processed_history += self.assistant_key + message['content'].strip() + '</s>'
            if message['role'].lower() == 'user':
                processed_history += self.user_key + message['content'].strip() + ' '

        return processed_history

    def forward( self, messages: List[Dict[str, str]] ) -> str:

        history = self._process_history( messages )
        prompt = history + self.assistant_key

        input_ids = self.tokenizer.encode( prompt, return_tensors="pt" ).to( self.config.mpt7B.device )

        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[ 1 ] + self.config.mpt7B.max_new_tokens,
            temperature=self.config.mpt7B.temperature,
            do_sample=not self.config.mpt7B.greedy_sampling,
            pad_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=self.config.mpt7B.repetition_penalty,
            top_k=self.config.mpt7B.top_k,
            top_p=self.config.mpt7B.top_p,
            stopping_criteria=StoppingCriteriaList( [ self.stop ] ),
        )

        generation = self.tokenizer.decode( output[ 0 ][ input_ids.shape[ 1 ]: ], skip_special_tokens=True )

        # Logging input and generation if debugging is active
        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation



if __name__ == "__main__":
    bittensor.utils.version_checking()
    Mpt7BMiner().run()