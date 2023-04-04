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
import asyncio
import argparse
import bittensor as bt

from copy import deepcopy

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
        parser.add_argument('--netuid', type=int , help = 'Prompting network netuid', default = 41 )
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_prompting_validator')
        parser.add_argument('--neuron.reward_model_name', type=str, help='GPTRewardModel name', default='Dahoas/gpt2-rm-static')
        parser.add_argument('--neuron.inference_topk', type=str, help='At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument('--neuron.training_topk', type=str, help='During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument('--neuron.epoch_length', type=str, help='During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )

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
        self.metagraph = self.subtensor.metagraph(21)
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
        self.reward_model = RewardModel( self.config.neuron.reward_model_name ).to(self.device)
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to(self.device)
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        self.history = []

    def uid_selector(
            self
    ):
        """
            Returns the uids of metagraph endpoint objs that are not equal to 0.0.0.0

            Returns:
                uids (list): The list of uids.
        """
        uids = []
        for endpoint in self.metagraph.endpoint_objs:
            if endpoint.ip != '0.0.0.0':
                print('ip |', endpoint.ip, '| uid |', endpoint.uid)
                uids.append(endpoint.uid)
        return uids
   
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
        if topk is None or topk == -1: topk = self.metagraph.n.item()

        # We run the gating network here to get the best uids
        # TODO( jason + joey ): this is not training properly.
        # probably want to port over joey's work on embedding based cosine similarity routing.
        # message --> torch.tensor([scores x num_uids])
        scores = self.gating_model( 
            message
        )

        # We query the topk best uids here using the inference topk as the limit.
        uids = scores.sort()[1][-topk:].tolist()
        
        _completions = self.dendrite_pool( 
            prompt = prompt,
            message = message, 
            uids = uids 
        )

        completions = [completion for completion in _completions if completion is not None]
        uids = [uid for completion, uid in zip(_completions, uids) if completion is not None]

        # We rank the completions based on the reward model.
        # TODO( carro ): this needs to be trained properly.
        rewards = self.reward_model.reward( 
            completions 
        )

        # import pdb; pdb.set_trace()

        # We backpropagate the rewards to the gating network.
        # TODO( jaons + joey ): ditto.


        self.gating_model.backward( 
            scores = scores[ uids ], 
            rewards = rewards 
        )

        # We call the backward pass to the miners.
        # TODO( jason ) write the proto + dendrite + synapse
        # to service these requests.
        # reward --> miner, then the miner optionally applies PPO.
        # self.dendrite_pool.backward(
        #     message = message,
        #     reward = rewards, # NOTE( jason ): I dont think this is a single scalar in practice. 
        #     uids = uids,
        # )

        # Save history for later computing of weights.
        # history [
        #     message, rewards # message = { 'role': 'user', 'text': 'hello world'}, rewards = [ 0.1, 0.2, 0.3, 0.4 ... x number of miners ]
        #     message, rewards
        #     ...
        #     ...
        # ] ---> weights
        self.history.append( 
            ( message, uids, rewards, scores[ uids ] )
        )
        #if self.config.logging.debug:
        for completion, reward, score in zip( completions, rewards, scores[ uids ] ):
            print( "Message: ", message )
            print( "Completion: ", completion )
            print( "Reward: ", reward )
            print( "Score: ", score )
            print( "------------------" )

        # We return the completion with the highest reward.
        return completions[ rewards.argmax(dim=0) ]
    
    # User queries here.
    def inference( self, message ):
        """Inference"""
        return self.forward( message, topk = self.config.neuron.inference_topk )

    def train(self):
        """ Training """
        last_epoch_block = self.subtensor.block
        while True:
            
            # TODO( robert ): Use prompting network here to generate inputs.
            # the prompting network should generate questions which cover our data distribution.
            message = self.forward( "ask me a random question about anything. Make the question very domain specific about science and language." )
            self.forward( message, topk = self.config.neuron.training_topk )
    
            # Set weights on epoch.
            if self.subtensor.block - last_epoch_block > self.subtensor.validator_epoch_length( self.config.netuid ) :
                last_epoch_block = self.subtensor.block

                # TODO( taco + joey + jason): we need to get a lot more intelligence here about how 
                # we translate the history into chain weights. Because we actually want to be
                # training the weights so that the E( topk( score ) )_B               
                # A: 5 5 5 5 5 5 5 5 5 5  = 5
                # B: 10 0 0 0 0 10 0 0 0 -- 20/8 = 3.5
                # C:  0 0 10 10 10 0 0 0 -- 30/8 = 3.9
                # D:  0 0 0 0 0 0 0 10 10 -- 20/8 = 3.5
                # gate_matrix = torch.zeros_like(weights)
                # gate_values = torch.nn.Softmax(dim=1)(topk.values)
                # self.gate_matrix = gate_matrix.scatter(dim=1, index=topk.indices, src=gate_values)

                # average_rewards = torch.zeros_like( self.metagraph.R )

                from collections import defaultdict
                scores_by_uid = defaultdict(list)
                for message, uids, rewards, scores in self.history:
                    for uid, reward in zip(uids, rewards):
                        scores_by_uid[uid].append(reward)
                        
                for k, v in scores_by_uid.values():
                    scores_by_uid[k] = sum(v) / len(v)
                weights = torch.zeros_like( self.metagraph.uids, dtype = torch.float32 )
                uids = sorted(list(scores_by_uid.keys()))
                rewards = [scores_by_uid[k] for k in uids]
                
                weights = weights.scatter(
                    index = torch.tensor( uids, dtype=torch.long ), 
                    values = torch.tensor( rewards, dtype = torch.float32 )
                )                    
                self.subtensor.set_weights(
                    uids = self.metagraph.uids,
                    weights = weights
                )
            
if __name__ == '__main__':
    neuron().train()

