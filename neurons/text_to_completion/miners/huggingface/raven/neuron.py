import os
import argparse
import bittensor
from typing import List, Dict
from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

class RavenMiner( bittensor.HuggingFaceMiner ):

    arg_prefix = 'raven'
    system_label = ""
    assistant_label = "Alice:"
    user_label = "Bob:"

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        super( RavenMiner, cls ).add_args( parser )
        parser.add_argument( '--raven.repo_id', type=str, default="BlinkDL/rwkv-4-raven", help='Repo id of model to load' )
        parser.add_argument( '--raven.tokenizer_path', type=str, required=True, help='Path to tokenizer json file' )
        parser.add_argument( '--raven.ctx_limit', type=int, help='Max context length for model input.', default=1536 )
        parser.add_argument( '--raven.jit_on', action='store_true', default=False, help='Whether to use Just-In-Time complication (JIT)' )
        parser.add_argument( '--raven.cuda_on', action='store_true', default=False, help='Whether to use CUDA kernel for seq mode (much faster). [Requires CUDA_HOME env_variable to be set]' )
        parser.add_argument( '--raven.strategy', type=str, default='cuda fp16i8 *8 -> cuda fp16', help='Strategy to use for RWKV model')

    def __init__(self):
        super( RavenMiner, self ).__init__()

    def load_model( self ):
        model_path = hf_hub_download( repo_id=self.config.raven.repo_id, filename=f"{self.config.raven.model_name}.pth" )
        model = RWKV( model=model_path, strategy=self.config.raven.strategy )
        return PIPELINE( model, self.config.raven.tokenizer_path )

    def load_tokenizer( self ):
        pass

        os.environ["RWKV_JIT_ON"]  = '1' if self.config.raven.jit_on else '0'
        os.environ["RWKV_CUDA_ON"] = '1' if self.config.raven.cuda_on else '0'

    def forward( self, messages: List[Dict[str, str]] ) -> str:
        history = self.process_history( messages )

        out_tokens = []
        out_last = 0
        generation = ''
        occurrence = {}
        state = None
        for i in range( self.config.raven.max_new_tokens ):
            tokens = self.config.raven.pad_tokens + self.model.encode( history ) if i == 0 else [token]
            
            out, state = self.model.model.forward(tokens, state)
            for n in occurrence:
                out[n] -= ( self.config.raven.repetition_penalty + occurrence[n] * self.config.raven.repetition_penalty )
            
            token = self.model.sample_logits( out, temperature=self.config.raven.temperature, top_p=self.config.raven.top_p )
            if token == 0: break # exit when 'endoftext'            
            
            out_tokens += [token]
            occurrence[token] = 1 + ( occurrence[token] if token in occurrence else 0 )
            
            tmp = self.model.decode( out_tokens[out_last:] )
            if ( '\ufffd' not in tmp ) and ( not tmp.endswith('\n') ):
                generation += tmp
                out_last = i + 1
            
            if '\n\n' in tmp: # exit when '\n\n'
                generation += tmp
                generation = generation.strip()
                break

        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    RavenMiner().run()

def test_miner( model ):
    prompt = """
    You are George Carlin.
    George Carlin is a comedian known for his witty, cutting, poignant observational comedy.
    He is also known for his social commentary, philosophy, and cutting remarks on religion.
    Write a joke about the following topic:
    """

    message = "who am I, really?"

    if prompt is not None: 
        roles = ['system', 'user']
        messages = [ prompt, message ]
    else:
        roles = ['user']
        messages = [ message ]

    messages = [{'role': role, 'content': message} for role, message in zip(roles, messages)]

    return model.forward( messages )

miner = RavenMiner()
print( test_miner(miner) )