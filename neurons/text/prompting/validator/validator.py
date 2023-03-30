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
from reward_impl import GPTRewardModel

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
        parser.add_argument('--netuid', type=int , help = 'Prompting network netuid', default = 11 )
        parser.add_argument('--neuron.name', type=str,
                        help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ',
                        default='prompting_validator')

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
        self.subtensor = bt.subtensor ( chain_endpoint='wss://test.finney.opentensor.ai', network="finney" )

        self.messages = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = self.subtensor.metagraph(11)
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
        self.reward_model = GPTRewardModel('Dahoas/gpt2-rm-static')
        self.reward_model.to(self.device)
        self.modules = [ bt.text_prompting( endpoint = endpoint, wallet = self.wallet ) for endpoint in self.metagraph.endpoint_objs ]

    async def query(self, user_message: str):
        coroutines = []
        for uid in self.metagraph.uids.tolist():
            coroutines.append(self.call_uid(uid, user_message))

        all_responses = await asyncio.gather(*coroutines)
        return all_responses
    
    async def call_uid(self, uid: int, user_message: str) -> str:
        endpoint = self.metagraph.endpoint_objs[uid]
        if endpoint.ip == "0.0.0.0" and endpoint.uid != 0: pass
        if endpoint.uid == 0:
            endpoint = bt.endpoint(
                version=bt.__version_as_int__,
                uid=0,
                ip="127.0.0.1",
                ip_type=4,
                port=8091,
                hotkey=self.wallet.hotkey.ss58_address,
                coldkey=self.wallet.coldkeypub.ss58_address,
                modality=0,
            )
        module = bt.text_prompting(endpoint=endpoint, wallet=self.wallet)
        response = await module.async_forward(roles=['user'], messages=[user_message], timeout=12)
        return response.response

    def run(self):
        while True:

            # Query the uid endpoint.
            async def call_uid( uid: int, user_message: str ) -> str:
                endpoint = self.metagraph.endpoint_objs[uid]
                if endpoint.ip == "0.0.0.0" and endpoint.uid != 0: pass
                if endpoint.uid == 0:
                    endpoint = bt.endpoint(
                        version=bt.__version_as_int__,
                        uid=0,
                        ip="127.0.0.1",
                        ip_type=4,
                        port=8091,
                        hotkey=self.wallet.hotkey.ss58_address,
                        coldkey=self.wallet.coldkeypub.ss58_address,
                        modality=0,
                    )
                module = bt.text_prompting( endpoint = endpoint, wallet = self.wallet )
                response = await module.async_forward(roles=['user'], messages=[user_message], timeout=12)
                return response.response
            
            # Async call the uids.
            async def query( user_message: str ):
                coroutines = []
                for uid in self.metagraph.uids.tolist():
                    coroutines.append( call_uid( uid, user_message ) )
                    
                all_responses = await asyncio.gather(*coroutines)
                return all_responses
            # Make queries
            user_message = input("User> ")
            responses_per_uid = asyncio.run(query(user_message))
            print(responses_per_uid)
            # Get the highest value response
            max_reward: float = -math.inf
            max_response: str = ""
            for uid, response in enumerate( responses_per_uid ):
                if response:
                    reward = self.reward_fn([response])
                    print(f"UID: {uid} | Reward: {reward} | Response: {response}")
                    if reward > max_reward:
                        max_reward = reward
                        max_response = response

            print("Bot> ", max_response)
            
    def reward_fn(self, samples):
        scores_list = []
        batch_size = 2
        for i in range(0, len(samples), batch_size):
            sub_samples = samples[i : i + batch_size]
            sub_samples = [
                "<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples
            ]
            encodings_dict = self.reward_model.tokenizer(
                sub_samples,
                truncation=True,
                max_length=550,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = encodings_dict["input_ids"].to(self.device)
            attn_masks = encodings_dict["attention_mask"].to(self.device)
            input_ids = input_ids.repeat(2, 1)
            attn_masks = attn_masks.repeat(2, 1)
            with torch.no_grad():
                sub_scores = self.reward_model(input_ids=input_ids, attention_mask=attn_masks)
            scores_list.append(sub_scores["chosen_end_scores"])
        scores = torch.cat(scores_list, dim=0)
        return scores



if __name__ == '__main__':
    neuron().run()

