# https://huggingface.co/mosaicml/mpt-7b-instruct

import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import AutoTokenizer, pipeline
from transformers import StoppingCriteria, StoppingCriteriaList


# TODO: Add Flast Attention dependencies and options
# TODO: Add Triton Attention dependencies and options

# Define a custom stopping criteria
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
        parser.add_argument( '--mpt7B.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        parser.add_argument( '--mpt7B.greedy_sampling', action='store_true', default=False, help='Whether to use greedy sampling or not (if not, uses multinomial sampling).')
        parser.add_argument( '--mpt7B.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( '--mpt7B.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "This is the most correct and relevant answer to the question, with a rating of 10." )
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

        bittensor.logging.info( 'Loading ' + str( self.config.mpt7B.model_name ) )
        self.pipe = pipeline(
            "text-generation",
            model="mosaicml/mpt-7b-instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device=self.config.mpt7B.device,
        )
        self.tokenizer = AutoTokenizer.from_pretrained( "EleutherAI/gpt-neox-20b" )
        self.stop = StopOnTokens( self.tokenizer.convert_tokens_to_ids(["<|endoftext|>"]) )
        bittensor.logging.info( 'Model loaded!' )

    def _process_history( self, history: List[Dict[str, str]] ) -> str:
        processed_history = ''

        if self.config.mpt7B.do_prompt_injection:
            processed_history += self.config.mpt7B.system_prompt

        for message in history:
            if message['role'].lower() == 'system':
                if not self.config.mpt7B.do_prompt_injection or message != history[0]:
                    processed_history += self.system_key + message['content'].strip() + ' '
            if message['role'].lower() == 'assistant':
                processed_history += self.assistant_key + message['content'].strip() + '</s>'
            if message['role'].lower() == 'user':
                processed_history += self.user_key + message['content'].strip() + ' '

        return processed_history
    
    def forward( self, messages: List[Dict[str, str]] ):
        history = self._process_history(messages)
        prompt = history + self.assistant_key

        input_ids = self.tokenizer(prompt, return_tensors="pt" ).input_ids
        input_ids = input_ids.to( self.pipe.model.device )

        gkw = {
            **{
                "input_ids": input_ids,
                "max_new_tokens": self.config.mpt7B.max_new_tokens,
                "temperature": self.config.mpt7B.temperature,
                "do_sample": not self.config.mpt7B.greedy_sampling,
                "top_p": self.config.mpt7B.top_p,
                "top_k": self.config.mpt7B.top_k,
                "repetition_penalty": self.config.mpt7B.repetition_penalty,
                "stopping_criteria": StoppingCriteriaList( [ self.stop ] ),
            },
        }
        output = self.pipe.model.generate(**gkw)
        generation = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation))
        return generation



if __name__ == "__main__":
    bittensor.utils.version_checking()
    Mpt7BMiner().run()