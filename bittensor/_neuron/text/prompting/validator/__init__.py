import torch
from torch import nn
import os
import sys

import bittensor
import argparse
import asyncio
import math
import re
import json

from typing import List, Tuple

from bittensor._neuron.text.prompting.validator.model_impl import PromptingValidator
from .reward_impl import GPTRewardModel

class neuron:
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
        bittensor.subtensor.add_args( parser )
        bittensor.metagraph.add_args( parser )
        bittensor.logging.add_args( parser )
        bittensor.dataset.add_args( parser )
        bittensor.wandb.add_args(parser)
        bittensor.prometheus.add_args( parser )
        return bittensor.config( parser )
    
    def __init__( self ):
        self.config = neuron.config()
        self.subtensor = bittensor.subtensor ( config = self.config )
        self.messages = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bittensor.wallet ( config = self.config )
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
        self.reward_model = GPTRewardModel('Dahoas/gptj-rm-IHP', 'EleutherAI/gpt-j-6B')
        self.reward_model.to(self.device)
        self.modules = [ bittensor.text_prompting( endpoint = endpoint, wallet = self.wallet ) for endpoint in self.metagraph.endpoints_objs ]

    def run(self):
        while True:

            # Query the uid endpoint.
            async def call_uid( uid: int, user_message: str ) -> str:
                endpoint = self.metagraph.endpoints_objs[uid]
                if endpoint.ip == "0.0.0.0" and endpoint.uid != 2: continue
                if endpoint.uid == 2:
                    endpoint = bittensor.endpoint(
                        version=bittensor.__version_as_int__,
                        uid=2,
                        ip="127.0.0.1",
                        ip_type=4,
                        port=8091,
                        hotkey=self.wallet.hotkey.ss58_address,
                        coldkey=self.wallet.coldkeypub.ss58_address,
                        modality=0,
                    )
                module = bittensor.text_prompting( endpoint = endpoint, wallet = self.wallet ) 
                return await module.async_forward( roles = ['user'], messages = [user_message], timeout=12 ).response
            
            # Async call the uids.
            async def query( user_message: str ):
                coroutines = []
                for uid in self.metagraph.uids.tolist():
                    coroutines.append( call_uid( uid, user_message ) )
                await asyncio.gather(*coroutines)

            # Make queries
            user_message = input("User> ")
            responses_per_uid = asyncio.run(query(user_message))

            # Get the highest value response
            max_reward: float = -math.inf
            max_response:str = ""
            for uid, response in enumerate( responses_per_uid ):
                reward = self.reward_fn([response])
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


