import os
from typing import List, Dict
import argparse
import bittensor
from huggingface_hub import hf_hub_download
from rwkv.model import RWKV
from rwkv.utils import PIPELINE


BASE_PROMPT = '''The following is a coherent verbose detailed conversation between <|user|> and an AI girl named <|bot|>.
<|user|>: Hi <|bot|>, Would you like to chat with me for a while?
<|bot|>: Hi <|user|>. Sure. What would you like to talk about? I'm listening.
'''


class RavenMiner( bittensor.BasePromptingMiner ):
    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( '--raven.model_name', type=str, default="RWKV-4-Raven-7B-v11x-Eng99%-Other1%-20230429-ctx8192", help='Name/path of model to load' )
        parser.add_argument( '--raven.repo_id', type=str, default="BlinkDL/rwkv-4-raven", help='Repo id of model to load' )
        parser.add_argument( '--raven.tokenizer_path', type=str, default="/home/jason/bittensor/neurons/text/prompting/miners/raven-rwkv/20B_tokenizer.json", help='Path to tokenizer json file' )
        parser.add_argument( '--raven.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( '--raven.ctx_limit', type=int, help='Max context length for model input.', default=1536 )
        parser.add_argument( '--raven.max_new_tokens', type=int, help='Max tokens for model output.', default=256 )
        parser.add_argument( '--raven.temperature', type=float, help='Sampling temperature of model', default=1.0 )
        parser.add_argument( '--raven.top_p', type=float, help='Top p sampling of model', default=0.0 )
        parser.add_argument( '--raven.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( '--raven.system_prompt', type=str, help='What prompt to replace the system prompt with', default=BASE_PROMPT )
        parser.add_argument( '--raven.jit_on', action='store_true', default=False, help='Whether to use Just-In-Time complication (JIT)' )
        parser.add_argument( '--raven.cuda_on', action='store_true', default=False, help='Whether to use CUDA kernel for seq mode (much faster). [Requires CUDA_HOME env_variable to be set]' )
        parser.add_argument( '--raven.strategy', type=str, default='cuda fp16i8 *8 -> cuda fp16', help='Strategy to use for RWKV model')
        parser.add_argument( '--raven.pad_tokens', type=int, default=[], nargs='+', help='A list of integers separated by spaces for the pad_tokens.')
        parser.add_argument( '--raven.repetition_penalty', type=float, default=0.2, help='Repetition penalty for RWKV model')

    def __init__(self):
        super( RavenMiner, self ).__init__()

        model_path = hf_hub_download( repo_id=self.config.raven.repo_id, filename=f"{self.config.raven.model_name}.pth" )
        self.model = RWKV( model=model_path, strategy=self.config.raven.strategy )
        self.pipeline = PIPELINE( self.model, self.config.raven.tokenizer_path )
        self.pad_tokens = self.config.raven.pad_tokens # [0] or [187] -> probably useful

        os.environ["RWKV_JIT_ON"]  = '1' if self.config.raven.jit_on else '0'
        os.environ["RWKV_CUDA_ON"] = '1' if self.config.raven.cuda_on else '0'

    def _process_history( self, history: List[Dict[str, str]] ) -> str:
        processed_history = ''
        for message in history:
            if message['role'] == 'system':
                processed_history += message['content'].strip() + ' '
            if message['role'] == 'Assistant':
                processed_history += 'Alice:' + message['content'].strip() + '</s>'
            if message['role'] == 'user':
                processed_history += 'Bob: ' + message['content'].strip() + ' '
        return processed_history

    def generate(self, query):
        out_tokens = []
        out_last = 0
        out_str = ''
        occurrence = {}
        state = None
        ctx = f'Bob:{query.strip()}\n\nAlice:'
        for i in range(self.config.raven.max_new_tokens):
            tokens = self.pad_tokens + self.pipeline.encode(ctx) if i == 0 else [token]
            
            out, state = self.pipeline.model.forward(tokens, state)
            for n in occurrence:
                out[n] -= (self.config.raven.repetition_penalty + occurrence[n] * self.config.raven.repetition_penalty)
            
            token = self.pipeline.sample_logits(out, temperature=self.config.raven.temperature, top_p=self.config.raven.top_p)
            if token == 0: break # exit when 'endoftext'            
            
            out_tokens += [token]
            occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
            
            tmp = self.pipeline.decode(out_tokens[out_last:])
            if ('\ufffd' not in tmp) and (not tmp.endswith('\n')): 
                out_str += tmp
                out_last = i + 1
            
            if '\n\n' in tmp: # exit when '\n\n'
                out_str += tmp
                out_str = out_str.strip()
                break

        print('\n' + '=' * 50)
        return out_str        

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self._process_history(messages)
        generation = self.generate(history)
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation))
        return generation


if __name__ == "__main__":
    bittensor.utils.version_checking()
    RavenMiner().run()