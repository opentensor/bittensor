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
import queue
import torch
import bittensor
import argparse
import bittensor as bt

from types import SimpleNamespace
from typing import List, Optional, Tuple, Dict
from reward import RewardModel
from gating import GatingModel
from transformers import AutoTokenizer
import transformers 
from dataclasses import dataclass, field, asdict

__default_question_prompt__ = '''
Ask me a random question about anything. Make the question very domain specific about science and language.
Do not include the answer in the question.
'''

__default_base_prompt__ = '''
You are designed to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics.
'''

class stats(dict):
    def create_entry(self, uid, hotkey):
        if uid in self.keys():
            bittensor.logging.warning(f'Entry for uid {uid} has already been created! {self[uid]}')
            return
        super().__setitem__(uid, stat(uid, hotkey))
    def update(self, uids, rewards, metagraph):
        for uid, reward in zip(uids, rewards):
            if uid not in self.keys():
                self.create_entry(uid, metagraph.hotkeys[uid])
            self[uid].update(reward)

    def scores(self):
        results = []
        for stat in self.values():
            results.append(stat.score)
        return results

    def reset_epoch(self):
        for stat in self.values():
            stat.reset_epoch()

    def data_as_dict(self):
        return { uid: stat.dict() for uid, stat in self.items()}
    
@dataclass
class stat:
    uid: int
    hotkey: str
    num_queries:int = 0 #number of queries
    success:int = 0 #number of successful response
    epoch_rewards: list = field(default_factory = lambda: [] ) #normalized rewards
    alpha: float = 0.1
    ema_reward: float = None
    ema_reward_with_none: int = None
    
    def update(self, reward) -> bool:
        self.num_queries += 1
            
        # reward = None when a peer fail to respond
        if reward != None and reward != 0: 
            self.success += 1
            self.epoch_rewards.append(reward)
            self.ema_reward = reward * (1 - self.alpha) + self.ema_reward * self.alpha if self.ema_reward != None else reward
            self.ema_reward_with_none = reward * (1 - self.alpha) + self.ema_reward_with_none * self.alpha if self.ema_reward_with_none != None else reward
        
        else: # assume reward to be 0 with a failing request 
            self.epoch_rewards.append(None)
            self.ema_reward_with_none = self.ema_reward_with_none * self.alpha if self.ema_reward_with_none != None else 0

        return True

    @property
    def avg_epoch_reward(self):
        if len(self.epoch_rewards) == 0:
            return 0
        rewards = [r for r in self.epoch_rewards if r != None]
        return sum(rewards) / len(rewards)
    
    @property
    def avg_epoch_reward_with_none(self):
        if len(self.epoch_rewards) == 0:
            return 0
        rewards = [r if r!= None else 0 for r in self.epoch_rewards]
        return sum(rewards) / len(rewards)

    @property
    def score(self):
        return self.avg_epoch_reward_with_none
    
    def reset_epoch(self):
        self.epoch_rewards = []

    def dict(self):
        return {k: v for k, v in asdict(self).items()}
        
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
        parser.add_argument( '--neuron.epoch_length', type = str, help = 'During training time, how many miners to we query for each batch based on scores from gating network.', default = 10 )
        parser.add_argument( '--neuron.reward_path', type = str, help = 'Path to reward model.', default = '~/.bittensor/reward_models' )
        parser.add_argument( '--neuron.max_history', type = int, help = 'Maximum number history values to store at any time.', default = 100 )
        parser.add_argument( '--neuron.base_timeout', type = int, help = 'Base timeout for all requests.', default = 1 )
        parser.add_argument('--neuron.track_hotkey_changes', action='store_true', help='If True, track hotkey changes.', default=False)
        

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

        #reward model
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

        #gating model
        self.gating_model = GatingModel( metagraph = self.metagraph, config = self.config ).to( self.device )
        
        self.dendrite_pool = bt.text_prompting_pool( metagraph = self.metagraph, wallet = self.wallet )
        
        self.neuron_stats = stats()
        self.neuron_hotkeys = []

    def get_weights( self ) -> Tuple[ torch.LongTensor, torch.FloatTensor ]:
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

        if len(self.neuron_stats.keys()) == 0:
            return torch.ones((self.metagraph.n)) / self.metagraph.n  
        # Process the raw weights to final_weights via subtensor limitations.
        uids, weights = bittensor.utils.weight_utils.process_weights_for_netuid(
            uids = torch.tensor(list(self.neuron_stats.keys())),
            weights = torch.tensor(self.neuron_stats.scores()),
            netuid = self.config.netuid,
            subtensor = self.subtensor,
            metagraph = self.metagraph
        )

        return uids, weights

    def metagraph_sync(self):
        r""" Syncing metagraph together with other metagraph-size related objects
        """
        old_hotkeys = self.neuron_hotkeys + [] if self.neuron_hotkeys else self.metagraph.hotkeys
        self.metagraph.sync( subtensor=self.subtensor, netuid=self.config.netuid)
        self.neuron_hotkeys = self.metagraph.hotkeys

        changed_hotkeys = []
        # === Reset neuron stats if uid got replaced
        for uid, old_hotkey in enumerate(old_hotkeys):
            if old_hotkey != self.neuron_hotkeys[uid]:
                if self.config.neuron.track_hotkey_changes:
                    block = self.subtensor.block
                    self.neuron_changes.setdefault(uid, {})  # [uid] -> dict() of blocks
                    self.neuron_changes[uid][block] = {'new_hotkey': self.neuron_hotkeys[uid], 'old_hotkey': old_hotkey}
                    if uid in self.neuron_stats:
                        self.neuron_changes[uid][block]['old_stats'] = self.neuron_stats[uid]

                if uid in self.neuron_stats:
                    del self.neuron_stats[uid]
                    changed_hotkeys += [uid]

        if len(changed_hotkeys):
            self.save()  # save neuron_stats, neuron_hotkeys, and neuron_changes to filesystem

    def save(self, path=None):
        r""" Save validated hotkeys and neuron_stats to filesystem. """
        try:
            if path is None:
                path = self.config.neuron.full_path

            state_dict = {
                'neuron_stats': self.neuron_stats.data_as_dict(),
                'neuron_hotkeys': self.neuron_hotkeys
            }

            if self.config.neuron.track_hotkey_changes:
                state_dict['neuron_changes'] = self.neuron_changes

            torch.save(state_dict, f'{path}/model.torch')
            bittensor.logging.success(prefix='Saved model', sufix=f'<blue>{path}/model.torch</blue>')

        except Exception as e:
            bittensor.logging.warning(f'Failed to save model with error: {e}')

    def forward(
            self, 
            message: str,
            topk: Optional[int] = None,
        ) -> SimpleNamespace:
        """
        Queries the network for a response to the passed message using a gating model to select the best uids.
        Trains the gating model based on the rewards calculated for the successful completions and passes rewards
        backward for potential PPO.

        Args:
            message (str): The message to query the network with.
            topk (Optional[int]): The number of uids to consider for the query. If None or -1, all uids will be considered.
                                 If provided, selects the top k uids based on the gating model scores.

        Returns:
            result (SimpleNamespace): A namespace containing the completion with the highest reward, message, uids,
                                      rewards, scores, and all completions.
        """
        bittensor.logging.info( 'forward()' )
        bittensor.logging.info( 'message', message.strip() )

        # Set `topk` to the number of items in `self.metagraph.n` if `topk` is not provided or is -1.
        # Find the available `uids` that are currently serving.
        # If `topk` is larger than the number of available `uids`, set `topk` to the number of available `uids`.
        available_uids = torch.tensor( [ uid for uid, ep in enumerate( self.metagraph.endpoint_objs ) if ep.is_serving ], dtype = torch.int64 ).to( self.device )
        if topk is None or topk == -1: topk = self.metagraph.n.item()
        if topk > len(available_uids): topk = len(available_uids)
        bittensor.logging.debug( 'topk', topk)
        if len( available_uids ) == 0: bittensor.logging.error('no available uids'); return None

        # We run the gating network here to get the best uids
        # Use the gating model to generate scores for each `uid`.
        scores = self.gating_model( message ).to( self.device )

        # TODO: add stochasticity by querying random uids with probability epsilon.
        bittensor.logging.debug( 'scores', scores )

        # Select the top `topk` `uids` based on the highest `scores`.
        # Use the selected `uids` to query the dendrite pool.
        # Print the `completions`.
        topk_uids = available_uids[ scores[ available_uids ].sort()[ 1 ][ -topk: ]]
        completions = self.dendrite_pool( 
            prompt = self.config.neuron.base_prompt, 
            message = message, 
            uids = topk_uids, 
            # timeout = float( self.config.neuron.base_timeout + self.config.neuron.length_timeout_multiplier * len( message ) ) #TODO: add timeout
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
            all_completions = completions
        )

        # Return the completion with the highest reward.
        bittensor.logging.debug( 'forward result', result )
        return result

    def update_stats(self, uids, rewards):
        normalized_rewards = torch.nn.functional.softmax( rewards.to( self.device ), dim=0 )
        self.neuron_stats.update(uids.tolist(), normalized_rewards.detach().tolist(), self.metagraph)

    # User queries here.
    def inference( self, message: str) -> str:
        """Inference"""
        bittensor.logging.info( 'inference()' )
        bittensor.logging.info( 'message', message.strip() )
        result = self.forward( message, topk = self.config.neuron.inference_topk )
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
            question = self.forward( self.config.neuron.question_prompt )
            if question == None: continue # no responses from network.
            
            # Ask the network to complete the random question, training the gating network.
            forward_result = self.forward( question.completion, topk = self.config.neuron.training_topk )
            if forward_result:
                self.update_stats(forward_result.uids, forward_result.rewards)
            # Check if enough epoch blocks have elapsed since the last epoch.
            if self.subtensor.block > last_epoch_block: # run every block. # > self.subtensor.validator_epoch_length( self.config.netuid ) :
                bittensor.logging.info( 'epoch()' )
                bittensor.logging.info( 'block', self.subtensor.block )

                # Update the last epoch block to the current epoch block.
                last_epoch_block = self.subtensor.block
                
                # Computes the average reward for each uid across non-zero values 
                # using the rewards history stored in the self.history list.

                uids, weights = self.get_weights()

                # Set the weights on chain via our subtensor connection.
                self.subtensor.set_weights(
                    wallet = self.wallet,
                    netuid = self.config.netuid,
                    uids = uids,
                    weights = weights,
                    wait_for_finalization = True,
                )

                self.save()
                
                self.neuron_stats.reset_epoch()

                self.metagraph_sync()

if __name__ == '__main__':
    bittensor.logging.info( 'neuron().train()' )
    neuron().train()