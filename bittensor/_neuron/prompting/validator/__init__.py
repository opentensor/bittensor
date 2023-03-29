import torch
from torch import nn
import os
import sys

import bittensor
import argparse
import re
import json

from typing import List, Tuple

from bittensor._neuron.prompting.validator.model_impl import PromptingValidator

class neuron:
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,
        model: 'bittensor.neurons.text.core_server.server' = None,
        netuid: int = None
    ):

        if config == None: config = neuron.config()

        subtensor = bittensor.subtensor ( config = config ) if subtensor == None else subtensor
        if config.subtensor.network != 'nakamoto' and config.netuid == None:
            config.netuid = subtensor.get_subnets()[0]

        # Verify subnet exists
        if config.subtensor.network != 'nakamoto' and not subtensor.subnet_exists( netuid = config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {config.netuid} does not exist[/red]")
            sys.exit(1)
        
        self.messages = []

    def __enter__(self):

        self.model = PromptingValidator(
            config = self.config,
            model_name = self.config.nucleus.model_name,
            min_tokens = self.config.nucleus.min_tokens,
            max_tokens = self.config.nucleus.max_tokens,
            temperature = self.config.nucleus.temperature,
            top_p = self.config.nucleus.top_p,
            logprobs = self.config.nucleus.logprobs,
            repetition_penalty = self.config.nucleus.repetition_penalty
        )


#{ "role": "human", "content": "I want to buy a new phone" },

#( instruction, response )

#[
# 'user:I want to buy a new phone',
#  'assistant:here is what i found: 
# 
# 
# 
# 1'
# ]

    def process_history(self, messages: List[dict]) -> Tuple[str, List[Tuple[str, str]]]:
        histories = []
        
        for message in messages:
            if message['role'] != 'assistant':
                instruction = message['content']
            else:
                response = message['content']
                histories.append((instruction, response))

        return histories

    def create_user_prompt(self, message):
        message = {
            "role": "user",
            "content": message
        }
        self.messages.append(json.dumps(message))

    # an input that creates a user prompt
    def debug_input(self):
        user_message = input("User> ")
        return user_message

        

    def run(self, prompt):
        with self:

            while True:
                user_message = self.debug_input()
                self.create_user_prompt(user_message)
                print(self.messages)




    @classmethod
    def check_config( cls, config: 'bittensor.Config' ):
        r""" Checks/validates the config namespace object.
        """
        PromptingValidator.check_config( config )
        bittensor.logging.check_config( config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.dendrite.check_config( config )
        bittensor.wandb.check_config( config )
        bittensor.prometheus.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        config.using_wandb = config.wandb.api_key != 'default'
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument('--netuid', type=int , help='Subnet netuid', default=11)

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        PromptingValidator.add_args( parser )    
    

        bittensor.wallet.add_args( parser )
        bittensor.dendrite.add_args( parser )
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )