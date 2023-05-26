import torch
import argparse
import bittensor
from typing import List, Dict
from transformers import T5Tokenizer, AutoModelForSeq2SeqLM

from base import HuggingFaceMiner

class FastChatT5Miner( HuggingFaceMiner ):

    arg_prefix: str = "fastchat_t5"
    assistant_label: str = "ASSISTANT:"
    user_label: str = "USER:"
    system_label: str = "SYSTEM:"

    def load_model( self ):
        bittensor.logging.info( 'Loading ' + str( self.config.fastchat_t5.model_name ) )
        model = AutoModelForSeq2SeqLM.from_pretrained( self.config.fastchat_t5.model_name, local_files_only=True, low_cpu_mem_usage=True, torch_dtype=torch.float16 )
        bittensor.logging.info( 'Model loaded!' )
        return model

    def load_tokenizer( self ):
        return T5Tokenizer.from_pretrained( self.config.fastchat_t5.model_name, local_files_only=True )

    def forward( self, messages: List[Dict[str, str]] ) -> str:
        history = self.process_history( messages )
        prompt = history + self.assistant_label
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.config.fastchat_t5.device)
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.fastchat_t5.max_new_tokens,
            temperature=self.config.fastchat_t5.temperature,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generation = self.tokenizer.decode( output[0], skip_special_tokens=True )

        bittensor.logging.debug( "Message: " + str( messages ) )
        bittensor.logging.debug( "Generation: " + str( generation ) )
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    FastChatT5Miner().run()

def test_messages():
    prompt = """
    You are George Carlin.
    George Carlin is a comedian known for his witty, cutting, poignant observational comedy.
    George is able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a real human, George Carlin is able to generate amazing humor and jokes based on the input he receives, allowing him to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
    """

    message = "who are you?"

    if prompt is not None: 
        roles = ['system', 'user']
        messages = [ prompt, message ]
    else:
        roles = ['user']
        messages = [ message ]

    messages = [{'role': role, 'content': message} for role, message in zip(roles, messages)]

    return messages

# miner = FastChatT5Miner()
# miner.forward( test_messages() )