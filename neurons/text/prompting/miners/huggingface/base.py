import bittensor
import argparse
from typing import List, Dict
from abc import ABC, abstractmethod

class HuggingFaceMiner( bittensor.BasePromptingMiner, ABC ):
    arg_prefix: str
    assistant_label: str
    user_label: str
    system_label: str

    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        pass

    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser ):
        parser.add_argument( f'--{cls.arg_prefix}.model_name', type=str, required=True, help='Name or path of model to load' )
        parser.add_argument( f'--{cls.arg_prefix}.device', type=str, help='Device to load model', default="cuda" )
        parser.add_argument( f'--{cls.arg_prefix}.max_new_tokens', type=int, help='Max tokens for model output.', default=256 )
        parser.add_argument( f'--{cls.arg_prefix}.temperature', type=float, help='Sampling temperature of model', default=0.5 )
        parser.add_argument( f'--{cls.arg_prefix}.do_sample', action='store_true', default=False, help='Whether to use multinomial sampling.' )
        parser.add_argument( f'--{cls.arg_prefix}.repetition_penalty', type=float, help='Repetition penalty for model', default=1.3 )
        parser.add_argument( f'--{cls.arg_prefix}.do_prompt_injection', action='store_true', default=False, help='Whether to use a custom "system" prompt instead of the one sent by bittensor.' )
        parser.add_argument( f'--{cls.arg_prefix}.system_prompt', type=str, help='What prompt to replace the system prompt with', default= "BEGINNING OF CONVERSATION: " )
        parser.add_argument( f'--{cls.arg_prefix}.repetition-penalty', type=float, default=1.1, help='Repetition penalty for greedy decoding. Between 1.0 and infinity. 1.0 means no penalty. Default: 1.0' )
        parser.add_argument( f'--{cls.arg_prefix}.top_p', type=float, default=0.9, help='Top-p (nucleus) sampling. Defaults to 1.0 (top-k sampling). Must be between 0.0 and 1.0.' )
        parser.add_argument( f'--{cls.arg_prefix}.top_k', type=int, default=0, help='Top-k sampling. Defaults to 0 (no top-k sampling). Must be between 0 and 1000.' )
        parser.add_argument( f'--{cls.arg_prefix}.load_in_8bit', type=bool, default=False, help='Load model in 8 bit precision')
        parser.add_argument( f'--{cls.arg_prefix}.pad_tokens', type=int, default=[], nargs='+', help='A list of integers separated by spaces for the pad_tokens.')

    def __init__(self):
        super( HuggingFaceMiner, self ).__init__()
        print( self.config )

        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

        # Device already configured if using pipieline. (i.e. Pipelines have no `.to()` method)
        if getattr(self.config, self.arg_prefix).device != "cpu" and 'pipeline' not in self.model.__class__.__name__.lower():
            self.model = self.model.to( getattr(self.config, self.arg_prefix).device )

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def load_tokenizer(self):
        ...

    @abstractmethod
    def forward(self, messages: List[Dict[str, str]], **kwargs) -> str:
        ...

    def process_history( self, history: List[Dict[str, str]] ) -> str:
        processed_history = ''

        if getattr(self.config, self.arg_prefix).do_prompt_injection:
            processed_history += getattr(self.config, self.arg_prefix).system_prompt

        for message in history:
            if message['role'] == 'system':
                if not getattr(self.config, self.arg_prefix).do_prompt_injection or message != history[0]:
                    processed_history += self.system_label + message['content'].strip() + ' '
            if message['role'] == 'assistant':
                processed_history += self.assistant_label + message['content'].strip() + '</s>'
            if message['role'] == 'user':
                processed_history += self.user_label + message['content'].strip() + ' '
        return processed_history
