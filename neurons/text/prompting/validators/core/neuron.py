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
import copy
import queue
import torch
import random
import bittensor
import argparse
import bittensor as bt

from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from reward import RewardModel
from gating import GatingModel
from transformers import AutoTokenizer
from dataclasses import dataclass, field, asdict

__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific. Do not include the answer in the question.
'''

__default_base_prompt__ = '''
You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
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
        # Netuid Arg
        parser.add_argument( '--netuid', type = int, help = 'Prompting network netuid', default = 41 )
        parser.add_argument( '--neuron.name', type = str, help = 'Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ', default = 'core_prompting_validator')
        parser.add_argument( '--neuron.base_prompt', type=str, help = 'Prompt injected before a question is completed by miners on the network', default = __default_base_prompt__ )
        parser.add_argument( '--neuron.question_prompt', type=str, help = 'Prompt used to generate questions from the network whicha are used to evaluate other miners.', default = __default_question_prompt__ )
        parser.add_argument( '--neuron.reward_model_name', type = str, help = 'GPTRewardModel name', default = 'Dahoas/gpt2-rm-static')
        parser.add_argument( '--neuron.length_timeout_multiplier', type = int, help = 'Base timeout for all requests.', default = 0.01 )
        parser.add_argument( '--neuron.inference_topk', type = str, help = 'At inference time, how many miners to we query and return the top rewarded.', default = 10 )
        parser.add_argument( '--neuron.training_topk', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument( '--neuron.reward_path', type = str, help = 'Path to reward model.', default = '~/.bittensor/reward_models' )
        parser.add_argument( '--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 1000 )

    @classmethod
    def config ( cls ):
        parser = argparse.ArgumentParser()    
        bt.wallet.add_args( parser )
        bt.subtensor.add_args( parser )
        bt.metagraph.add_args( parser )
        bt.logging.add_args( parser )
        GatingModel.add_args( parser )
        cls.add_args( parser )
        return bt.config( parser )
    
    def __init__( self, config=None ):
        self.config = config if config is not None else neuron.config()
        self.check_config(self.config)
        bt.logging( config = self.config )
        self.config.neuron.reward_path = os.path.expanduser(self.config.neuron.reward_path)
        if not os.path.exists( self.config.neuron.full_path):
            os.makedirs(self.config.neuron.full_path, exist_ok=True)
        if not os.path.exists( self.config.neuron.reward_path + '/hf_ckpt.pt' ):
            os.makedirs(self.config.neuron.reward_path, exist_ok=True)
            os.system(
                f"wget -O {self.config.neuron.reward_path + '/hf_ckpt.pt'} \
                https://huggingface.co/Dahoas/gptj-rm-static/resolve/main/hf_ckpt.pt"
            )

        self.subtensor = bt.subtensor ( config = self.config )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.wallet = bt.wallet ( config = self.config )
        self.metagraph = self.subtensor.metagraph( self.config.netuid )
        self.wallet.create_if_non_existent()
        self.wallet.reregister( subtensor = self.subtensor, netuid = self.config.netuid )
        self.uid = self.wallet.get_uid( subtensor = self.subtensor, netuid = self.config.netuid )
        self.tokenizer = AutoTokenizer.from_pretrained( 'EleutherAI/gpt-j-6b' )

        # Reward model
        self.reward_model = RewardModel('EleutherAI/gpt-j-6b')
        for fpath in os.listdir( self.config.neuron.reward_path ):
            if fpath.endswith(".pt") or fpath.endswith(".bin"):
                checkpoint = os.path.join( self.config.neuron.reward_path, fpath )
                break

        ckpt_state = torch.load( checkpoint )
        self.reward_model.load_state_dict( ckpt_state )
        self.reward_model.eval()
        self.reward_model.requires_grad_( False )
        self.reward_model.to( self.device )

        # Init the gating model which learns which miners to select for each query.
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to( self.device )
        # Denddrite pool for querying the network.
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        # History of forward events.
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
        bittensor.logging.info( 'compute_weights()' )

        # Return zeros weights if there is no history.
        if self.history.qsize() == 0: 
            bittensor.logging.warning( 'No history to compute weights returning all ones.' )
            return torch.ones((self.metagraph.n)) / self.metagraph.n

        # Iterate over all events in the `history` and perform a moving average of the normalized rewards.
        alpha = 0.01
        last_hotkeys = None
        moving_averaged_scores = torch.zeros((self.metagraph.n)).to( self.device )
        for event in self.history.queue:    
            # First we normalize the rewards with a softmax.
            normalized_rewards = torch.nn.functional.softmax( event.rewards.to( self.device ), dim=0 )
            # We scatter the normalized onto the moving averaged scores (updating them but not changing the source)
            scattered_rewards = moving_averaged_scores.scatter(0, event.uids.to( self.device ), normalized_rewards.to( self.device ) )
            # We now perform a moving average of the scattered rewards.
            moving_averaged_scores = alpha * moving_averaged_scores + ( 1 - alpha ) * scattered_rewards
            bittensor.logging.debug( 'normalized_rewards', normalized_rewards )
            bittensor.logging.debug( 'scattered_rewards', scattered_rewards )
            bittensor.logging.debug( 'moving_averaged_scores', moving_averaged_scores )

            # If the hotkeys have changed, reset the moving averaged scores for the new hotkeys.
            if last_hotkeys is None:
                for uid, hotkey in enumerate( event.hotkeys ):
                    if hotkey != last_hotkeys[ uid ]:
                        moving_averaged_scores[ uid ] = 0
            # Update the last hotkeys.
            last_hotkeys = event.hotkeys

        # Calculate the average reward for each uid across non-zero values.
        # Replace any NaN values with 0.
        raw_weights = torch.nn.functional.normalize( moving_averaged_scores, p=1, dim=0 )
        bittensor.logging.debug( 'raw_weights', raw_weights )
        bittensor.logging.debug( 'top10 values', raw_weights.sort()[0] )
        bittensor.logging.debug( 'top10 uids', raw_weights.sort()[1] )
     
        # Process the raw weights to final_weights via subtensor limitations.
        processed_weight_uids, processed_weights = bittensor.utils.weight_utils.process_weights_for_netuid(
            weights = raw_weights,
            netuid = self.config.netuid,
            subtensor = self.subtensor,
            metagraph = self.metagraph
        )
        bittensor.logging.debug( 'processed_weights', processed_weights )
        bittensor.logging.debug( 'processed_weight_uids', processed_weight_uids )
        return processed_weight_uids, processed_weights

    def forward(
            self, 
            message: str,
            topk: Optional[int] = None,
            random_sample_uids: Optional[ bool ] = False,
            train_gating_model: Optional[ bool ] = False,
        ) -> SimpleNamespace:
        """
        Queries the network for a response to the passed message using a gating model to select the best uids.
        Trains the gating model based on the rewards calculated for the successful completions and passes rewards
        backward for potential PPO.

        Args:
            message (str): 
                The message to query the network with.
            topk (Optional[int]): 
                The number of uids to consider for the query. If None or -1, all uids will be considered.
                If provided, selects the top k uids based on the gating model scores.
            random_sample_uids( bool, default = False ):
                If True, randomly samples the uids to query rather than using topk.
            train_gating_model ( bool, default = False ):
                If True, trains the gating model based on the rewards calculated for the successful completions.
        Returns:
            result (SimpleNamespace): 
                A namespace containing the completion with the highest reward, message, uids,
                rewards, scores, and all completions.
        """
        bittensor.logging.info( 'forward()' )
        bittensor.logging.info( 'message', message.strip() )

        # Set `topk` to the number of items in `self.metagraph.n` if `topk` is not provided or is -1.
        # Find the available `uids` that are currently serving.
        # If `topk` is larger than the number of available `uids`, set `topk` to the number of available `uids`.
        available_uids = torch.tensor( [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ], dtype = torch.int64 ).to( self.device )
        if topk is None or topk == -1: topk = self.metagraph.n.item()
        if topk > len( available_uids ): topk = len( available_uids )
        bittensor.logging.debug( 'topk', topk)
        if len( available_uids ) == 0: bittensor.logging.error('no available uids'); return None

        # We run the gating network here to get the best uids
        # Use the gating model to generate scores for each `uid`.
        scores = self.gating_model( message ).to( self.device )
        bittensor.logging.debug( 'scores', scores )

        # Select the top `topk` `uids` based on the highest `scores`.
        # Use the selected `uids` to query the dendrite pool.
        # Print the `completions`.
        if random_sample_uids:
            topk_uids = torch.tensor( random.sample( available_uids.tolist(), topk ), dtype = torch.int64 ).to( self.device )
        else:
            topk_uids = available_uids[ scores[ available_uids ].sort()[ 1 ][ -topk: ]]
        completions = self.dendrite_pool( 
            prompt = self.config.neuron.base_prompt, 
            message = message, 
            uids = topk_uids, 
        )
        bittensor.logging.debug( 'topk_uids', topk_uids )
        bittensor.logging.debug( 'completions', completions )

        # Filter out any `None` `completions`.
        successful_uids = torch.tensor([uid for uid, completion in list(zip(topk_uids, completions)) if completion is not None and completion.response is not None and len(completion.response) > 10], dtype=torch.int64).to(self.device)
        successful_completions = [completion.response for completion in completions if completion is not None and completion.response is not None and len(completion.response) > 10]
        bittensor.logging.debug( 'successful_uids', successful_uids )
        bittensor.logging.debug( 'successful_completions', successful_completions )
        if len( successful_completions ) == 0: bittensor.logging.error('no successful completions'); return None

        # Calculate the rewards for the successful `completions` using the reward model.
        # Print the rewards for all `uids`.
        rewards = self.reward_model.reward( successful_completions ).to( self.device )
        bittensor.logging.debug( 'rewards', rewards )

        # Train the gating model using the scores and rewards of the successful `completions`.
        if train_gating_model:
            self.gating_model.backward( scores = scores[ successful_uids ], rewards = rewards )
            bittensor.logging.debug( 'Apply backward to gating model' )

        # Pass rewards backward for potential PPO.
        self.dendrite_pool.backward( 
            prompt = self.config.neuron.base_prompt, 
            message = message, 
            completions = successful_completions,
            rewards = rewards,
            uids = successful_uids, 
        )
        bittensor.logging.debug( 'Applied backward to network.' )

        # Save the query history in a `result` object.
        # Return the `completion` with the highest reward.
        result = SimpleNamespace( 
            completion = successful_completions[ rewards.argmax( dim = 0 ) ],
            message = message,  
            uids = successful_uids,
            rewards = rewards,
            scores = scores,
            all_completions = completions,
            hotkeys = copy.deepcopy( self.metagraph.hotkeys ),
        )
        self.history.push( result )

        # Return the completion with the highest reward.
        bittensor.logging.debug( 'forward result', result )
        return result

    # User queries here.
    def inference( self, message: str) -> str:
        """Inference"""
        bittensor.logging.info( 'inference()' )
        bittensor.logging.info( 'message', message.strip() )
        result = self.forward( 
            message = message, 
            topk = self.config.neuron.inference_topk,
            random_sample_uids = False,
            train_gating_model = True
        )
        if result == None: return "Failed"
        else: return result.completion

    def train( self ):
        """ Training 
            The function uses an infinite loop to repeatedly generate a random question, 
            ask the network to complete the question, and train the gating network using 
            the question and the resulting completions.
        """
        # Store the current epoch block number for comparison later.
        last_epoch_block = self.subtensor.block + 100
        
        # Start an infinite loop for training.
        while True:
            
            # Query the network for a random question.
            question = self.forward( 
                self.config.neuron.question_prompt,
                topk = self.config.neuron.training_topk,
                random_sample_uids = True,
                train_gating_model = True,
            )
            if question == None: continue # no responses from network.
            
            # Ask the network to complete the random question, training the gating network.
            self.forward( 
                message = question.completion, 
                topk = self.config.neuron.training_topk,
                random_sample_uids = True,
                train_gating_model = True,
            )

            # Resync metagraph before returning. (sync every 15 min or ~75 blocks)
            if last_epoch_block % 75 == 0:
                self.metagraph = self.metagraph.sync(netuid=self.config.netuid, subtensor=self.subtensor)

            # Check if enough epoch blocks have elapsed since the last epoch.
            if self.subtensor.block - last_epoch_block > self.subtensor.validator_epoch_length( self.config.netuid ): 
                bittensor.logging.info( 'epoch()' )
                bittensor.logging.info( 'block', self.subtensor.block )

                # Synce the metagraph.
                self.metagraph = self.subtensor.metagraph( self.config.netuid )

                # Update the last epoch block to the current epoch block.
                last_epoch_block = self.subtensor.block
                
                # Computes the average reward for each uid across non-zero values 
                # using the rewards history stored in the self.history list.
                uids, weights = self.compute_weights()
                bittensor.logging.info( 'weights', weights )

                # Set the weights on chain via our subtensor connection.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = uids,
                    weights = weights,
                    wait_for_finalization = True,
                )

if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().train()
