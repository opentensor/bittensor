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
import warnings
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList


class TextGenerationPipeline:
    def __init__(
        self,
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_auth_token=None,
        device="cuda",
    ) -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
            max_seq_len=10240, 
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            use_auth_token=use_auth_token,
        )
        if tokenizer.pad_token_id is None:
            warnings.warn(
                "pad_token_id is not set for the tokenizer. Using eos_token_id as pad_token_id."
            )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer
        self.model.eval()
        self.model.to(device=device, dtype=torch_dtype)


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
        parser.add_argument( '--mpt7B.model_name', type=str, default='mosaicml/mpt-7b-instruct', help='Name/path of model to load' )
        parser.add_argument( '--mpt7B.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--mpt7B.max_new_tokens', type=int, help='Max tokens for model output.', default=512 )
        parser.add_argument( '--mpt7B.temperature', type=float, help='Sampling temperature of model', default=0.8 )
        parser.add_argument( '--mpt7B.greedy_sampling', action='store_true', default=False, help='Whether to use greedy sampling or not (if not, uses multinomial sampling).' )
        parser.add_argument( '--mpt7B.no_repeat_ngram_size', type=int, default=3, help='If set to int > 0, all ngrams of size no_repeat_ngram_size can only occur once.' )
        parser.add_argument( '--mpt7B.top_p', type=float, default=0.95, help='Top-p (nucleus) sampling. Defaults to 1.0 (top-k sampling). Must be between 0.0 and 1.0.' )
        parser.add_argument( '--mpt7B.top_k', type=int, default=50, help='Top-k sampling. Defaults to 0 (no top-k sampling). Must be between 0 and 1000.' )
        parser.add_argument( '--mpt7B.repitition_penalty', type=float, default=1.02, help='Repetition penalty for greedy decoding. Between 1.0 and infinity. 1.0 means no penalty. Default: 1.0' )

    def __init__( self ):
        super( Mpt7BMiner, self ).__init__()
        print ( self.config )

        bittensor.logging.info( 'Loading ' + str( self.config.mpt7B.model_name ) )
        self.pipe = TextGenerationPipeline(
            "mosaicml/mpt-7b-storywriter",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=self.config.mpt7B.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained( "EleutherAI/gpt-neox-20b" )
        self.stop = StopOnTokens( self.tokenizer.convert_tokens_to_ids( [ "<|endoftext|>" ] ) )
        bittensor.logging.info( 'Model loaded!' )

    def _process_history( self, history: List[Dict[str, str]] ) -> str:
        processed_history = ''

        for message in history:
            if message['role'].lower() == 'system':
                if message != history[0]:
                    processed_history += message['content'].strip() + ' '
            if message['role'].lower() == 'assistant':
                processed_history += message['content'].strip() + '</s>'
            if message['role'].lower() == 'user':
                processed_history += message['content'].strip() + ' '

        return processed_history
    
    def forward( self, messages: List[Dict[str, str]] ):
        history = self._process_history( messages )
        prompt = history

        input_ids = self.tokenizer( prompt, return_tensors="pt" ).input_ids
        input_ids = input_ids.to( self.pipe.model.device )

        gkw = {
            **{
                "input_ids": input_ids,
                "max_new_tokens": self.config.mpt7B.max_new_tokens,
                "temperature": self.config.mpt7B.temperature,
                "do_sample": not self.config.mpt7B.greedy_sampling,            
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "top_p": self.config.mpt7B.top_p,
                "top_k": self.config.mpt7B.top_k,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "stopping_criteria": StoppingCriteriaList( [ self.stop ] ),
            },
        }
        output = self.pipe.model.generate( **gkw )
        generation = self.tokenizer.decode( output[ 0 ][ input_ids.shape[ 1 ]: ], skip_special_tokens=True )

        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation



if __name__ == "__main__":
    bittensor.utils.version_checking()
    Mpt7BMiner().run()