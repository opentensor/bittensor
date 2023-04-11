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
import queue
import argparse
import bittensor as bt

from typing import List, Optional, Tuple

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
    
    def __init__( self, config = None ):
        self.config = config if config is not None else neuron.config()
        bt.logging( config = config )
        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )  
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        self.history = queue.Queue( maxsize = self.config.neuron.max_history )

    def compute_weights( self ) -> Tuple[ torch.LongTensor, torch.FloatTensor ]:
        """
            Computes the average reward for each uid across non-zero values 
            using the rewards history stored in the self.history list.

            Returns:
                uids ( torch.LongTensor, shape = (n) ): 
                    Uid to set weights on.
                weights ( torch.FloatTensor, shape = (n) ): 
                    The weights for each uid.
        """
        bt.logging.info( 'compute_weights()' )

        # Return zeros weights if there is no history.
        if self.history.qsize() == 0: 
            bt.logging.warning( 'No history to compute weights returning all ones.' )
            return torch.ones((self.metagraph.n)) / self.metagraph.n

        # Averages the rewards for each uid across non-zero values.
        rewards = []

        # Iterate over all events in the `history` list.
        for scoring in self.history.queue:
            # Normalize the rewards for the current event using softmax normalization.
            normalized_rewards = torch.nn.functional.softmax( scoring.to( self.device ), dim=0 )
            bt.logging.debug( 'normalized_rewards', normalized_rewards )

            # Use the `uids` of the current event to scatter the normalized rewards
            # into a zero-initialized tensor with the same shape as `self.metagraph.n`.
            scattered_rewards = torch.zeros((self.metagraph.n)).to( self.device ).scatter(0, self.metagraph.uids.to( self.device ), normalized_rewards.to( self.device ) )
            bt.logging.debug( 'scattered_rewards', scattered_rewards )

            # Append the scattered rewards to the `rewards` list.
            rewards.append( scattered_rewards )

        # Stack the scattered rewards tensors along the second dimension.
        rewards = torch.stack( rewards, 1 ).to( self.device )
        bt.logging.debug( 'rewards', rewards )

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nan_to_num( rewards.sum(1) / (rewards != 0).sum(1), 0 )
        bt.logging.debug( 'raw_weights', raw_weights )
        bt.logging.debug( 'top10 values', raw_weights.sort()[0] )
        bt.logging.debug( 'top10 uids', raw_weights.sort()[1] )
     
        # Process the raw weights to final_weights via subtensor limitations.
        processed_weight_uids, processed_weights = bt.utils.weight_utils.process_weights_for_netuid(
            weights = raw_weights,
            netuid = self.config.netuid,
            subtensor = self.subtensor,
            metagraph = self.metagraph
        )
        bt.logging.debug( 'processed_weights', processed_weights )
        bt.logging.debug( 'processed_weight_uids', processed_weight_uids )
        return processed_weight_uids, processed_weights


    def train(self):
        last_epoch_block = self.subtensor.block
        all_serving_uids = [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ]
        while True:

            # Generate question.
            bt.logging.debug('Question ---------------')
            question = None
            while True:
                question_responses = self.dendrite_pool( 
                    prompt = default_prompt, 
                    message = self.config.question_prompt, 
                    uids = all_serving_uids 
                )
                bt.logging.success( 'question_responses', question_responses )
                for qresp in question_responses:
                    if qresp.response != None:
                        question = qresp.response
                        bt.logging.success( 'question', question )
                if question != None:
                    bt.logging.success( 'Found question' )
                    break
                else:
                    bt.logging.warning( 'No question try again.' )
                    continue


            # Generate completions.
            bt.logging.debug('Completion ---------------')
            completion_prompt = self.config.completion_prompt.format( question )
            completions = []
            completion_uids = []
            number_of_completions = 2
            while True:
                completion_responses = self.dendrite_pool( 
                    prompt = default_prompt, 
                    message = completion_prompt, 
                    uids = all_serving_uids
                )
                bt.logging.success( 'completion_responses', completion_responses )
                for uid, cresp in list(zip( all_serving_uids, completion_responses ) ):
                    if cresp.response != None:
                        completions.append( cresp.response )
                        completion_uids.append( uid )
                        bt.logging.success( 'completion', cresp.response )
                if len(completions) > number_of_completions:
                    bt.logging.success( 'Found enough completions.' )
                    break
                else:
                    bt.logging.warning( 'Not enough completions try again.' )
                    continue


            # Generate evaluations
            bt.logging.debug('Evaluation ---------------')
            evaluation_prompt = extend_evaluation_prompt_with_question_criteria_and_completions( 
                criteria  = self.config.evaluation_prompt,
                question = question, 
                completions = completions 
            )
            evaluations = []
            evaluation_uids = []
            number_of_evaluations = 3
            while True:
                evaluation_responses = self.dendrite_pool( 
                    prompt = default_prompt, 
                    message = evaluation_prompt, 
                    uids = all_serving_uids
                )
                bt.logging.success( 'evaluation_responses', evaluation_responses )
                for uid, eresp in list(zip( all_serving_uids, evaluation_responses ) ):
                    if eresp.response != None:
                        evaluations.append( eresp.response )
                        evaluation_uids.append( uid )
                        bt.logging.success( 'evaluation', eresp.response )
                if len(evaluations) > number_of_evaluations:
                    bt.logging.success( 'Found enough evaluations.' )
                    break
                else:
                    bt.logging.warning( 'Not enough evaluations try again.' )
                    continue

            # Calculate reward
            bt.logging.debug('Scoring ---------------')
            for evaluation in evaluations:
                winner_uid = get_winner_from_evaluation( completion_uids, evaluation )
                bt.logging.debug('winner_uid', winner_uid)
                scoring  = torch.nn.functional.one_hot( torch.tensor( winner_uid ), num_classes =  self.metagraph.n ).float()
                bt.logging.debug('scoring', scoring )
                self.history.put( scoring )

            # Check if enough epoch blocks have elapsed since the last epoch.
            if self.subtensor.block > last_epoch_block: # run every block. # > self.subtensor.validator_epoch_length( self.config.netuid ) :
                bt.logging.info( 'epoch()' )
                bt.logging.info( 'block', self.subtensor.block )

                # Update the last epoch block to the current epoch block.
                last_epoch_block = self.subtensor.block
                
                # Computes the average reward for each uid across non-zero values 
                # using the rewards history stored in the self.history list.
                _, weights = self.compute_weights()
                bt.logging.info( 'top10 uids', weights.sort()[1][-10:] )
                bt.logging.info( 'top10 weights', weights.sort()[0][-10:] )

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

