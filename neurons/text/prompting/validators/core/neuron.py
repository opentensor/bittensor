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

import os
import torch
import argparse
import bittensor as bt

from types import SimpleNamespace
from collections import defaultdict
from typing import List, Optional
from reward import RewardModel
from gating import GatingModel

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

no_response_text = "no responce"

class neuron:
    @classmethod
    def check_config( cls, config: 'bt.Config' ):
        r""" Checks/validates the config namespace object.
        """
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
        parser.add_argument('--netuid', type = int , help = 'Prompting network netuid', default = 41 )
        parser.add_argument('--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument('--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument('--neuron.inference_topk', type = str, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument('--neuron.training_topk', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument('--neuron.epoch_length', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.metagraph.add_args( parser )
        bt.logging.add_args( parser )
        GatingModel.add_args( parser )
        return bt.config( parser )
    
    def __init__( self, config=None ):
        self.config = config if config is not None else neuron.config()
        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
        self.reward_model = RewardModel( self.config.neuron.reward_model_name ).to(self.device)
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to(self.device)
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        self.history = []

    def complute_weights( self ) -> torch.FloatTensor:
        """ Computes valdiation weights from history and clears history.
            Returns:
                weights ( torch.FloatTensor, shape = (n) ): 
                    The weights for each uid.
        """
        # Averages the rewards for each uid across non-zero values.
        rewards = []
        for event in self.history:
            normalized_rewards = torch.nn.functional.normalize( event.rewards, p=1, dim=0 )
            rewards.append( torch.zeros((self.metagraph.n)).scatter(0, event.uids, normalized_rewards ) )
        rewards = torch.cat( rewards, 0 )
        rewards = torch.stack( rewards, 1) 
        return torch.nan_to_num( rewards.sum(1) / (rewards != 0).sum(1), 0 )
   
    def forward( 
            self, 
            message: str,
            topk: Optional[int] = None,
        ):
        """ Inference is called by clients seeking the outputs of the model
            We use the gating network to determine the best models to query 
            Optionally we use the reward model to train the gating network.

            Args: 
                message (str): The message to query the network with.
        """
        print ('forward --------------------' )
        if topk is None or topk == -1: topk = self.metagraph.n.item()
        available_uids = [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ]
        if topk > len(available_uids): topk = len(available_uids)
        print ('topk', topk)

        # We run the gating network here to get the best uids
        scores = self.gating_model( message )
        print ('scores', scores)

        # We query the topk best uids here using the inference topk as the limit.
        topk_uids = available_uids[ scores[ available_uids ].sort()[ 1 ][ -topk: ]]
        print ('topk_uids', topk_uids)

        # We query the topk available uids.
        completions = self.dendrite_pool( prompt = prompt, message = message, uids = topk_uids )
        print ('completions', completions)

        # Filter out the None completions.
        successful_uids = [ uid for uid, completion in list( zip( topk_uids, completions ) ) if completion is not None ]
        successful_completions = [ completions[ uid ] for uid in successful_uids ]
        print ('successful_uids', successful_uids)

        # Rank the completions based on the reward model filling the rewards vector.
        rewards = torch.zeros( ( self.metagraph.n.item() ) )
        rewards[ successful_uids ] = self.reward_model.reward( successful_completions )
        print ('rewards', rewards)

        # We train the gating network here.
        self.gating_model.backward( scores = scores[ successful_uids ], rewards = rewards[ successful_uids ] )

        # Save the query history.
        result = SimpleNamespace( 
            completion = completions[ rewards.argmax( dim = 0 ) ],
            message = message,  
            uids = topk_uids,
            rewards = rewards,
            scores = scores[ topk_uids ],
            all_completions = completions
        )
        self.history.append( result )

        # Return the completion with the highest reward.
        return result
    
    # User queries here.
    def inference( self, message ):
        """Inference"""
        return self.forward( message, topk = self.config.neuron.inference_topk ).completion

    def train(self):
        """ Training """
        last_epoch_block = self.subtensor.block
        while True:

            # Query the network for a question            
            question = self.forward( "ask me a random question about anything. Make the question very domain specific about science and language." ).completion

            # Ask the questions to the network, training the gating network.
            self.forward( question, topk = self.config.neuron.training_topk )
    
            # Set weights on epoch.
            if self.subtensor.block - last_epoch_block > self.subtensor.validator_epoch_length( self.config.netuid ) :
                last_epoch_block = self.subtensor.block
                weights = self.complute_weights()          
                self.subtensor.set_weights(
                    uids = self.metagraph.uids,
                    weights = weights,
                )
            
if __name__ == '__main__':
    neuron().train()

