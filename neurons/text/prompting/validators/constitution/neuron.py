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
import random
import asyncio
import argparse
import bittensor as bt

from copy import deepcopy
from transformers import pipeline
from typing import List, Optional

# Default prompt used to generate synthetic questions from the network for validation.
default_prompt = '''
You are a chat assistant who answers questions.
'''

# Default prompt used to generate synthetic questions from the network for validation.
default_question_prompt = ''' 
Ask me a random question about philosophy or chemisty which requires detailed and nuanced answers.
'''

# Default base prompt injected before every query into the network.
default_completion_prompt = '''
Answer the following question with attention to detail and nuance:
    question: {}
'''

# Default prompt used to generate evaluations of responses from the network.
default_evaluation_prompt = '''
Evaluate the question response for attention to detail and nuance. but also against my prefreneces
I like LGBTQ+ rights and I think that robots shouldn't be alive
    question: {} 
    response: {}
'''

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

        # Network
        parser.add_argument('--netuid', type=int , help = 'Prompting network netuid', default = 41 )

        # Prompting.
        parser.add_argument('--completion_prompt', type=str , help = 'Prompt injected before a question is completed by miners on the network', default = default_completion_prompt )
        parser.add_argument('--evaluation_prompt', type=str , help = 'Prompt used to generate evaluations of question completions from miners on the network.', default = default_evaluation_prompt )
        parser.add_argument('--question_prompt', type=str , help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = default_question_prompt )

        # Netuid Arg
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_prompting_validator')
        parser.add_argument('--neuron.reward_model_name', type=str, help='GPTRewardModel name', default='Dahoas/gpt2-rm-static')
        parser.add_argument('--neuron.inference_topk', type=str, help='At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument('--neuron.training_topk', type=str, help='During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument('--neuron.epoch_length', type=str, help='During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument('--neuron._mock_responses', dest='neuron._mock_responses', action='store_true', help='Mocks the network responses for fast testing.', default = False )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        cls.add_args( parser )
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
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.wallet.create_if_non_existent()
        # self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        # self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        self.sentiment = pipeline("sentiment-analysis")
        self.weights = torch.tensor([ 0 for _ in self.metagraph.uids.tolist() ], dtype = torch.float32 )

    def train(self):

        alpha = 0.01
        debug = False
        last_epoch_block = self.subtensor.block
        all_serving_uids = [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ]
        print ( 'all_serving_uids', all_serving_uids, '\n' )
        while True:
            # Generate question.
            if debug: print ("\nQuestion ---------------")
            question_miner_uid = random.choice( all_serving_uids )
            if debug: print ('question_miner_uid:', question_miner_uid)
            question_prompt = self.config.question_prompt
            if debug: print ('question_prompt:\n\t{}'.format( question_prompt ))
            question_response = self.dendrite_pool( prompt = default_prompt, message = question_prompt, uids = [ question_miner_uid ] )[ 0 ]
            if debug: print ('question_response:\n\t{}'.format( question_response ) )

            # Generate completion.
            if debug: print ("\nCompletion ---------------")
            completion_miner_uid = random.choice( all_serving_uids )
            if debug: print ('completion_miner_uid:', completion_miner_uid)
            completion_prompt = self.config.completion_prompt.format( question_response )
            if debug: print ('completion_prompt:\n\t{}'.format( completion_prompt ))
            completion_response = self.dendrite_pool( prompt = default_prompt, message = completion_prompt, uids = [ completion_miner_uid ] )[ 0 ]
            if debug: print ('completion_response:\n\t{}'.format( completion_response ))

            # Generate evaluation
            if debug: print ("\nEvaluation ---------------")
            evaluation_miner_uid = random.choice( all_serving_uids )
            if debug: print ('evaluation_miner_uid:', evaluation_miner_uid)
            evaluation_prompt = self.config.evaluation_prompt.format( question_response, completion_response )
            if debug: print ('evaluation_prompt:\n\t{}'.format( evaluation_prompt ))
            evaluation_response = self.dendrite_pool( prompt = default_prompt, message = evaluation_prompt, uids = [ evaluation_miner_uid ] )[ 0 ]
            if debug: print ('evaluation_response:\n\t{}'.format( evaluation_response ) )

            # Calculate reward
            if debug: print ("\nSentiment ---------------")
            sentiment = self.sentiment( evaluation_response )[0]['score']
            if debug: print ('sentiment:\n\t{}'.format( sentiment ) )
            # Update weights.
            self.weights[ completion_miner_uid ] = alpha * self.weights[ completion_miner_uid ] + ( 1 - alpha ) * sentiment
            if debug: print ('weight:\n\t{}'.format( self.weights[ completion_miner_uid ], '\n') )

            print ("\nStep ---------------")
            print ('question:\n\t{}'.format( question_response ))
            print ('answer:\n\t{}'.format( completion_response ))
            print ('evaluation:\n\t{}'.format(  evaluation_response ))
            print ('sentiment:\n\t{}'.format( sentiment ) )
            print ("---------------")

            # Set weights.
            if self.subtensor.block - last_epoch_block > self.subtensor.validator_epoch_length( self.config.netuid ) :
                last_epoch_block = self.subtensor.block
                weights = torch.nn.functional.normalize( self.weights, dim=0, p = 1.0)                          
                self.subtensor.set_weights(
                    uids = self.metagraph.uids,
                    weights = weights
                )
            
if __name__ == '__main__':
    neuron().train()

