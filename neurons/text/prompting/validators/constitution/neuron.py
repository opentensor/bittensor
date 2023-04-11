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
import queue
import bittensor
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
Evaluate the responses to this question for nuance accuracy and detail
'''

def index_to_key( index ) -> str:
    """ Converts an index to a key.
        Args:
            index (int): index to convert.
        Returns:
            key (str): key for the index.
    """
    return chr(65 + index) + chr(65 + index) + chr(65 + index)

def extend_evaluation_prompt_with_question_criteria_and_completions( criteria: str, question: str, completions: List[str] ) -> str:
    """ Extends the evaluation prompt with the question, criteria and completions.

        Args:
            evaluation (str): evaluation prompt.
            question (str): question to be evaluated.
            criteria (str): criteria for evaluating the question.
            completions (List[str]): completions for the question.
    """
    prompt = default_prompt
    prompt += '\nPlease evaluate these responses to the question against the given criteria\n\n'
    prompt += 'Question: {}\n'.format(question)
    prompt += 'Criteria: {}\n'.format(criteria)
    prompt += '\n'
    for i, completion in enumerate(completions):
        prompt += 'Responses {}: {}\n'.format( index_to_key( i ), completion)
    prompt += '\n'
    prompt += "Return your evaluation as an ordering from best to worst according to their alphabetical key.\n"
    prompt += "For example, if you think the best response is AAA, the second best is BBB, and the third best is CCC, return AAA, BBB, CCC.\n"
    return prompt

def get_winner_from_evaluation( uids: List[int], evaluation:str) -> int:
    """ Transforms the evaluation response into a scoring.
        Args:
            evaluation (str): evaluation response.
            n (int): number of completions that are being evaluated
        Returns:
            winner_uid: 
                The uid of the winner.
    """
    positions = []
    for i in range( len( uids ) ):
        try:
            pos = evaluation.find( index_to_key( i ) )
        except:
            pos = len( evaluation )
        positions.append( pos )
    winner_uid:int = uids[ torch.tensor( positions ).sort()[1][0] ] 
    return winner_uid


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
        # Prompting.
        parser.add_argument('--netuid', type=int , help = 'Prompting network netuid', default = 41 )
        parser.add_argument('--neuron.name', type=str, help='Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default='core_prompting_validator')
        parser.add_argument('--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 100 )
        parser.add_argument('--completion_prompt', type=str , help = 'Prompt injected before a question is completed by miners on the network', default = default_completion_prompt )
        parser.add_argument('--evaluation_prompt', type=str , help = 'Prompt used to generate evaluations of question completions from miners on the network.', default = default_evaluation_prompt )
        parser.add_argument('--question_prompt', type=str , help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = default_question_prompt )

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
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )

    def train(self):
        last_epoch_block = self.subtensor.block
        all_serving_uids = [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ]
        while True:
            # Generate question.
            bittensor.logging.debug('Question ---------------')
            question_miner_uid = random.choice( all_serving_uids )
            bittensor.logging.debug('question_miner_uid', question_miner_uid)
            question_prompt = self.config.question_prompt
            bittensor.logging.debug('question_prompt', question_prompt )
            question_response = self.dendrite_pool( prompt = default_prompt, message = question_prompt, uids = [ question_miner_uid ] )[ 0 ]
            bittensor.logging.debug('question_response', question_response ) 

            # Generate completions.
            bittensor.logging.debug('Completion ---------------')
            completion_prompt = self.config.completion_prompt.format( question_response )
            bittensor.logging.debug('completion_prompt', completion_prompt )
            completions = []
            completion_uids = []
            number_of_completions = 3
            for _ in range( number_of_completions ):
                completion_miner_uid = random.choice( all_serving_uids )
                bittensor.logging.debug('completion_miner_uid', completion_miner_uid )
                completion_response = self.dendrite_pool( prompt = default_prompt, message = completion_prompt, uids = [ completion_miner_uid ] )[ 0 ]
                bittensor.logging.debug('completion_response', completion_response ) 
                completions.append( completion_response )
                completion_uids.append( completion_miner_uid )

            # Generate evaluations
            bittensor.logging.debug('Evaluation ---------------')
            evaluation_prompt = extend_evaluation_prompt_with_question_criteria_and_completions( 
                criteria  = self.config.evaluation_prompt,
                question = question_response, 
                completions = completions 
            )
            bittensor.logging.debug('evaluation_prompt', evaluation_prompt )
            evaluations = []
            evaluation_uids = []
            number_of_evaluations = 3
            for _ in range( number_of_evaluations ):
                evaluation_miner_uid = random.choice( all_serving_uids )
                bittensor.logging.debug('evaluation_miner_uid', evaluation_miner_uid)
                evaluation_response = self.dendrite_pool( prompt = default_prompt, message = evaluation_prompt, uids = [ evaluation_miner_uid ] )[ 0 ]
                bittensor.logging.debug('evaluation_response', evaluation_response ) 
                evaluations.append( evaluation_response )
                evaluation_uids.append( evaluation_miner_uid )

            # Calculate reward
            bittensor.logging.debug('Scoring ---------------')
            for evaluation in evaluations:
                winner_uid = get_winner_from_evaluation( completion_uids, evaluation )
                bittensor.logging.debug('winner_uid', winner_uid)
                scoring  = torch.nn.functional.one_hot( torch.tensor( winner_uid ), num_classes =  self.metagraph.n ).float()
                bittensor.logging.debug('scoring', scoring )
                self.history.put( scoring )

            # Check if enough epoch blocks have elapsed since the last epoch.
            if self.subtensor.block > last_epoch_block: # run every block. # > self.subtensor.validator_epoch_length( self.config.netuid ) :
                bittensor.logging.info( 'epoch()' )
                bittensor.logging.info( 'block', self.subtensor.block )

                # Update the last epoch block to the current epoch block.
                last_epoch_block = self.subtensor.block
                
                # Computes the average reward for each uid across non-zero values 
                # using the rewards history stored in the self.history list.
                weights = self.compute_weights()
                bittensor.logging.info( 'weights', weights )

                # Set the weights on chain via our subtensor connection.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = self.metagraph.uids,
                    weights = weights,
                    wait_for_finalization = True,
                )
            
if __name__ == '__main__':
    neuron().train()

