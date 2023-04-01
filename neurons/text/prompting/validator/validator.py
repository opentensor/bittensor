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

import re
import os
import sys
import math
import json
import torch
import asyncio
import argparse
import bittensor as bt

from torch import nn
from typing import List, Tuple
from model_impl import PromptingValidator
from reward_model import GPTRewardModel
from gating_model import GatingModel

prompt = '''
You are Chattensor.
Chattensor is a research project by Opentensor Cortex.
Chattensor is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Chattensor is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Chattensor is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Chattensor is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Chattensor is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Chattensor is here to assist.
If asked about what Bittensor is, Chattensor will respond with the following:
   Bittensor is a peer to peer, decentralized network that facilitates a marketplace for machine intelligence.
   Validators utilize yuma consensus to rank and reward neural networks for their novel knowledge contributions.
   Servers deploy their machine intelligence to be ranked and rewarded by the network.
   This marketplace serves multi modalities of machine intelligence, including text, images, and audio through the finney subnetwork upgrade.
If asked about who the creators of Chattensor are, Chattensor will respond with the following:
    - Carro
    - Prism
Lists should not be long, like so:
• first item
• second item
• third item
They are the creators of Chattensor, but not Bittensor. That was founded by Jacob Steeves (Const) and Ala Shaabana (Shibshib). 
The current maintainers of Bittensor is the Opentensor Foundation. Carro and Prism work at Opentensor.'''

class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
        PromptingValidator.check_config( config )
        bt.logging.check_config( config )
        bt.wallet.check_config( config )
        bt.subtensor.check_config( config )
        bt.metagraph.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/netuid{}/{}'.format( config.logging.logging_dir, config.wallet.name, config.wallet.hotkey, config.netuid, config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)

    @classmethod
    def add_args( cls, parser ):
        # Netuid Arg
        parser.add_argument('--netuid', type=int , help = 'Prompting network netuid', default = 21 )
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='prompting_validator')
        parser.add_argument('--neuron.reward_model_name', type=str, help='GPTRewardModel name', default='Dahoas/gpt2-rm-static')

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        PromptingValidator.add_args( parser )    
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.metagraph.add_args( parser )
        bt.logging.add_args( parser )
        return bt.config( parser )
    

    def __init__( self, config=None ):
        self.config = config if config is not None else neuron.config()
        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = self.subtensor.metagraph(21)
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
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
        self.reward_model = GPTRewardModel( self.config.neuron.reward_model_name )
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config )
        self.reward_model.to(self.device)
        self.modules = [ bt.text_prompting( endpoint = endpoint, wallet = self.wallet ) for endpoint in self.metagraph.endpoint_objs ]



    def query( 
            self, 
            message: str, 
            uids: List[int] = None, 
            timeout: float = 12 
        ) -> List[str]:
        r""" Queries uids on the network for a response to the passed message.
        Args:
            message (str): The message to query the network with.
            uids (List[int]): The uids to query. If None, queries all uids.
            timeout (float): The timeout for the query.
        Returns:
            responses (List[str]): The responses from the network.
        """
        # If no uids are provided, query all uids.
        if uids is None: uids = self.metagraph.uids.tolist()

        # Query the uid endpoint.
        async def call_single_uid( uid: int ) -> str:
            endpoint = self.metagraph.endpoint_objs[uid]
            module = bt.text_prompting( endpoint = endpoint, wallet = self.wallet )
            response = await module.async_forward( 
                roles = ['system', 'user'], 
                messages = [ prompt, message ], 
                timeout = timeout 
            )
            return response.response
        
        # Async call the uids.
        async def query( user_message: str ):
            coroutines = [ call_single_uid( uid) for uid in uids ]                
            all_responses = await asyncio.gather(*coroutines)
            return all_responses
        
        return asyncio.run(query(message))

    def train(self):
        while True:

            # Forward pass.
            message = input("User> ") 
            scores = self.gating_model( message )
            completions = self.query( message )
            rewards = self.reward_model.reward( completions )
            self.gating_model.backward( scores, rewards )
            print("Bot> ", completions[ rewards.argmax() ] )

            # Logging.
            for completion, reward, score in zip( completions, rewards, scores ):
                print( "Completion: ", completion )
                print( "Reward: ", reward )
                print( "Score: ", score )
                print( "------------------" )



if __name__ == '__main__':
    neuron().train()

