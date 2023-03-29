import torch
from torch import nn
import os
import sys

import bittensor
import argparse
import re
import json

from typing import List, Tuple

from bittensor._neuron.text.prompting.validator.model_impl import PromptingValidator
from .reward_impl import GPTRewardModel

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
        self.config = config
        subtensor = bittensor.subtensor ( config = config ) if subtensor == None else subtensor
        if config.subtensor.network != 'nakamoto' and config.netuid == None:
            config.netuid = subtensor.get_subnets()[0]

        # Verify subnet exists
        if config.subtensor.network != 'nakamoto' and not subtensor.subnet_exists( netuid = config.netuid ):
            bittensor.__console__.print(f"[red]Subnet {config.netuid} does not exist[/red]")
            sys.exit(1)
        
        self.messages = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bittensor.wallet ( config = self.config ) if wallet == None else wallet
        self.subtensor = subtensor
        self.metagraph = self.subtensor.metagraph(11)


    def sync_metagraph(self):
        self.metagraph.sync(netuid=self.config.netuid, subtensor=self.subtensor).save()

    def __exit__(self, exc_type, exc_value, traceback):
        print(exc_type, exc_value, traceback)

    def __enter__(self):
        config = self.config

        self.wallet.create()
        self.wallet.reregister( subtensor = self.subtensor, netuid=self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid=self.config.netuid )  

        self.model = PromptingValidator(
            config = config,
            model_name = config.nucleus.model_name,
            min_tokens = config.nucleus.min_tokens,
            max_tokens = config.nucleus.max_tokens,
            temperature = config.nucleus.temperature,
            top_p = config.nucleus.top_p,
            logprobs = config.nucleus.logprobs,
            repetition_penalty = config.nucleus.repetition_penalty
        )

        self.reward_model = GPTRewardModel('Dahoas/gptj-rm-IHP', 'EleutherAI/gpt-j-6B')
        self.reward_model.to(self.device)

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

    def create_user_prompt(self, message):
        message = {
            "role": "user",
            "content": message
        }
        return (json.dumps(message))

    # an input that creates a user prompt
    def debug_input(self):
        user_message = input("User> ")
        return user_message

        

    def run(self):
        with self:
            while True:
                user_message = self.debug_input()
                # message = self.create_user_prompt(user_message)
                responses = []
                values = {}
                for endpoint in (self.metagraph.endpoint_objs):
                    try:

                        # check if endpoint.ip is not 0.0.0.0
                        if endpoint.ip == "0.0.0.0" and endpoint.uid != 2:
                            continue
                            
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

                        # print(endpoint)
                        module = bittensor.text_prompting( endpoint = endpoint, wallet = self.wallet )
                        response = module.forward(
                            roles = ['user'],
                            messages = [user_message],
                            timeout=12
                        )
                        if response.response:
                            responses.append({ 'uid': endpoint.uid, 'response': response.response })
                            continue
                    except Exception as e:
                        print(e)
                        continue
                
                # print(responses)
                for response in responses:
                    value = self.reward_fn([response['response']])
                    values[response['response']] = value
                    # print(value)
                    
                # sort by value
                sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)

                # get the highest value response
                highest_value = sorted_values[0][0]

                # replace the prompt in the highest value response
                highest_value = highest_value.replace(user_message, "")

                print('Bittensor> ', highest_value)

                ## debug

                print('values: ', values)
                print('sorted_values: ', sorted_values)
                print('responses: ', responses)


    


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